import numpy as np
import pickle as pkl
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from Dataset import Dataset
from ActivityRecognitionModel import ActivityRecognitionModel
import time


class Train:
    # Train constants
    GPU_ID = 0
    EPOCH_NUM = 100
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 10
    TEST_RATE = 0.25
    OVERLAP_SIZE = 4

    # TASK: set pkl file path
    FRAMED_DATA_FILE_PATH = './output/lstm_frame/*.pkl'

    def __init__(self):
        self.xp = np
        self.features, self.label, self.actions = self.load_framed_data()
        self.train_data, self.test_data = self.shuffle_data()
        self.model = self.set_model()
        self.to_gpu()
        self.optimizer = self.set_optimizer()

    def load_framed_data(self):
        print('Framed Data Loading ...')
        with open(self.FRAMED_DATA_FILE_PATH, 'rb') as f:
            dataset = pkl.load(f)
        print('data number: ', str(len(dataset['label'])))
        return dataset['features'], dataset['label'], dataset['actions_dict']

    def shuffle_data(self):
        """
        任意の割合でランダムに訓練データとテストデータの tuple を返却する
        :return tuple (train, test):
            (
                ([features, features, ...], [label, label, ...]),
                ([features, features, ...], [label, label, ...])
            )
        """
        self._print_title('shuffle data to train and test')

        # index（データ数配列）をランダムに並び替え、2つにデータを分ける
        test_num = int(len(self.label) * self.TEST_RATE)
        indexes = np.array(range(len(self.label)))
        np.random.shuffle(indexes)
        train_indexes = indexes[test_num:]
        test_indexes = indexes[:test_num]

        # 空の train, test データを生成
        train_data, train_label = [], []
        test_data, test_label = [], []

        # 各々作成したランダムな index からデータを追加
        print('>>> create train data')
        for i in train_indexes:
            train_data += [self.features[i]]
            train_label += [self.label[i]]
        print('>>> create test data')
        for i in test_indexes:
            test_data += [self.features[i]]
            test_label += [self.label[i]]

        # メモリ開放のため
        del self.features
        del self.label

        return (train_data, train_label), (test_data, test_label)

    def set_model(self) -> ActivityRecognitionModel:
        self._print_title('set model:')
        # model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, self.dataset.get_output_size())
        model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, len(self.actions))
        return model

    def to_gpu(self):
        self._print_title('use gpu')
        cuda.check_cuda_available()
        cuda.get_device(self.GPU_ID).use()
        self.xp = cuda.cupy

        # 全データを cuda に突っ込むと GPU がだいぶ厳しいみたい。。
        # self.train_data = self.xp.asarray(self.train_data[0]), self.xp.asarray(self.train_data[1])
        # self.test_data = self.xp.asarray(self.test_data[0]), self.xp.asarray(self.test_data[1])
        self.train_data = np.array(self.train_data[0]), np.array(self.train_data[1])
        self.test_data = np.array(self.test_data[0]), np.array(self.test_data[1])
        self.model.to_gpu(self.GPU_ID)

    def set_optimizer(self):
        self._print_title('set optimizer')
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)
        return optimizer

    def random_batches(self, data) -> list:
        """
        train_data をランダムにバッチサイズずつに分割したデータ・ラベルのリストを返却する
        :return:
            [
                # （全データ/バッチサイズ）回繰り返し（各要素が対応）
                (
                    # バッチサイズの時系列画像特徴データ
                    [
                        [[feature], [feature], ... [feature]],
                        [[feature], [feature], ... [feature]],
                        ....
                    ],
                    # バッチサイズの時系列画像特徴データ
                    [
                        [label],
                        [label],
                        ...
                    ]
                ),
                ....
            ]
        """
        # batches = []
        length = len(data[1])
        index = np.arange(length, dtype=np.int32)
        np.random.shuffle(index)
        for i in range(0, length, self.BACH_SIZE):
            batch_indexes = index[i:i+self.BACH_SIZE]
            # batches.append((data[0][batch_indexes], data[1][batch_indexes]))
            yield (data[0][batch_indexes], data[1][batch_indexes])
        # return batches

    def main(self):
        self._print_title('main')



        for epoch in range(self.EPOCH_NUM):
            self._print_title('epoch: {}'.format(epoch + 1))
            # TODO: loss , accuracy 辺りが間違ってそう
            self.train()
            self.test()

    def train(self):
        sum_loss = 0
        sum_acc = 0
        cnt = 0
        loop_count = int(len(self.train_data[1])/self.BACH_SIZE)
        for i, (feature_batch, label_batch) in enumerate(self.random_batches(self.train_data)):
            self.model.cleargrads()
            loss, acc = self.model(feature_batch, label_batch)
            loss.backward()
            # loss.unchain_backward()  # TODO: これは必要なのだろうか??
            self.optimizer.update()
            sum_loss += loss
            sum_acc += acc
            cnt += 1
            if i % 50 == 0:
                print('{} / {} loss: {} accuracy: {}'.format(i + 1, loop_count, sum_loss/cnt, sum_acc/cnt))
        print('<<< train loss: {} accuracy: {}'.format(sum_loss/cnt, sum_acc/cnt))

    def test(self):
        sum_loss = 0
        sum_acc = 0
        size = 0
        cnt = 0
        for i, (feature_batch, label_batch) in enumerate(self.random_batches(self.test_data)):
            loss, acc = self.model(feature_batch, label_batch)
            sum_loss += loss
            sum_acc += acc
            size += len(label_batch)
            cnt += 1
        print('<<< test loss: {} accuracy: {}'.format(sum_loss/cnt, sum_acc/cnt))

    @staticmethod
    def _print_title(s: str):
        """
        現在の処理内容とデバッグしたいデータ（配列）を渡す）
        :param str s:
        :return:
        """
        print()
        print('<<<<', s, '>>>>')


if __name__ == '__main__':
    Train.FRAMED_DATA_FILE_PATH = './output/lstm_frame/20180913_111307.pkl'
    train = Train()
    # train.to_gpu(0)
    train.main()
