import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from Dataset import Dataset
from ActivityRecognitionModel import ActivityRecognitionModel


class Train:
    GPU_ID = 0
    EPOCH_NUM = 20
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 10
    TEST_RATE = 0.25

    def __init__(self):
        self.xp = np
        self.dataset = self.set_dataset()
        self.train_data, self.test_data = self.dataset.shuffle_to_train_test(self.TEST_RATE)
        self.model = self.set_model()
        self.to_gpu()
        self.optimizer = self.set_optimizer()

    def set_dataset(self) -> Dataset:
        self._print_title('set dataset')
        dataset = Dataset()
        return dataset

    def set_model(self) -> ActivityRecognitionModel:
        self._print_title('set model:')
        model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, self.dataset.get_output_size())
        return model

    def set_optimizer(self):
        self._print_title('set optimizer')
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)
        return optimizer

    def to_gpu(self):
        self._print_title('use gpu')
        cuda.check_cuda_available()
        cuda.get_device(self.GPU_ID).use()
        self.xp = cuda.cupy
        self.train_data = self.xp.asarray(self.train_data[0]), self.xp.asarray(self.train_data[1])
        self.test_data = self.xp.asarray(self.test_data[0]), self.xp.asarray(self.test_data[1])
        self.model.to_gpu(self.GPU_ID)

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
        batches = []
        length = len(data[1])
        index = np.arange(length, dtype=np.int32)
        np.random.shuffle(index)
        for i in range(0, length, self.BACH_SIZE):
            batch_indexes = index[i:i+self.BACH_SIZE]
            batches.append((data[0][batch_indexes], data[1][batch_indexes]))
        return batches

    def main(self):
        self._print_title('main')
        for epoch in range(self.EPOCH_NUM):
            self._print_title('epoch: {}'.format(epoch + 1))
            self.train()
            self.test()

    def train(self):
        batches = self.random_batches(self.train_data)
        sum_loss = 0
        sum_acc = 0
        cnt = 0
        for i, (feature_batch, label_batch) in enumerate(batches):
            self.model.cleargrads()
            loss, acc = self.model(feature_batch, label_batch)
            loss.backward()
            # loss.unchain_backward()  # TODO: これは必要なのだろうか??
            self.optimizer.update()
            sum_loss += loss
            sum_acc += acc
            cnt += 1
            if i % 5 == 0:
                print('{} / {} loss: {} accuracy: {}'.format(i + 1, len(batches), sum_loss/cnt, sum_acc/cnt))
        print('<<< train loss: {} accuracy: {}'.format(sum_loss/cnt, sum_acc/cnt))

    def test(self):
        sum_loss = 0
        sum_acc = 0
        size = 0
        cnt = 0
        batches = self.random_batches(self.test_data)
        for i, (feature_batch, label_batch) in enumerate(batches):
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
    train = Train()
    # train.to_gpu(0)
    train.main()
