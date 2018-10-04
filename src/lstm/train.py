import numpy as np
import pickle as pkl
import chainer
from chainer import cuda, datasets, iterators, optimizers, serializers, functions as F
import time
from datetime import datetime
import os
import sys
import cupy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from ActivityRecognitionModel import ActivityRecognitionModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Train:
    # Train constants
    OUTPUT_BASE = './output/model/'

    # setup
    GPU_DEVICE = 0
    EPOCH_NUM = 1000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 10
    TEST_RATE = 0.25
    OVERLAP_SIZE = 4

    def __init__(self, framed_cnn_pkl_path: str):
        self.xp = np
        self.framed_cnn_pkl_path = framed_cnn_pkl_path
        self.train_data, self.test_data, self.actions = None, None, None
        self.model, self.optimizer, self.save_dir = None, None, None
        self.save_dir = ''

        self.first_frame_data = []
        self.first_frame_label = []

    def _prepare(self):
        self._load_framed_data()
        self._set_model()
        self._set_optimizer()
        self._make_save_dir()
        self._dump_actions_data()
        self._to_gpu()

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            # self.xp = cuda.cupy
            self.xp = cupy
            self.model.to_gpu(self.GPU_DEVICE)
            print()
            print(type(self.xp))

    def _load_framed_data(self):
        self._print_title('Framed Data Loading ...')
        with open(self.framed_cnn_pkl_path, 'rb') as f:
            dataset = pkl.load(f)
        print('data number: ', str(len(dataset['label'])))
        self.train_data, self.test_data = self._shuffle_data(dataset['features'], dataset['label'])
        self.actions = dataset['actions_dict']

    def _shuffle_data(self, loaded_features, loaded_label):
        """
        任意の割合でランダムに訓練データとテストデータの tuple を返却する
        :return tuple (_train, test):
            (
                ([features, features, ...], [label, label, ...]),
                ([features, features, ...], [label, label, ...])
            )
        """
        self._print_title('shuffle data to train and test')

        # index（データ数配列）をランダムに並び替え、2つにデータを分ける
        test_num = int(len(loaded_label) * self.TEST_RATE)
        indexes = np.array(range(len(loaded_label)))
        np.random.shuffle(indexes)
        train_indexes = indexes[test_num:]
        test_indexes = indexes[:test_num]

        # 空の _train, test データを生成
        train_data, train_label = [], []
        test_data, test_label = [], []

        # TASK: データが増えたらこの辺で MemoryError 起こすかも。。
        # 各々作成したランダムな index からデータを追加
        print('>>> create train data')
        for i in train_indexes:
            train_data += [loaded_features[i]]
            train_label += [loaded_label[i]]
        print('>>> create test data')
        for i in test_indexes:
            test_data += [loaded_features[i]]
            test_label += [loaded_label[i]]

        # メモリ開放のため
        del loaded_features
        del loaded_label

        print('>>> train_data length: %s, train_label length: %s' % (len(train_data), len(train_label)))
        print('>>> test_data length: %s, test_label length: %s' % (len(test_data), len(test_label)))

        return (np.array(train_data), np.array(train_label)), (np.array(test_data), np.array(test_label))

    def _set_model(self) -> ActivityRecognitionModel:
        self._print_title('set model:')
        self.model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, len(self.actions))

    def _set_optimizer(self) -> optimizers:
        self._print_title('set optimizer')
        optimizer = optimizers.Adam()
        self.optimizer = optimizer.setup(self.model)

    def _make_save_dir(self):
        save_dir = self.OUTPUT_BASE + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir)
        self.save_dir = save_dir

    def _dump_actions_data(self):
        print('dump to actions in: ' + self.save_dir)
        path = self.save_dir + '/actions.pkl'

        with open(path, 'wb') as f:
            pkl.dump(self.actions, f, pkl.HIGHEST_PROTOCOL)

    def main(self):
        self._print_title('main')
        self._prepare()
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        
        for epoch in range(self.EPOCH_NUM):
            self._print_title('epoch: {}'.format(epoch + 1))
            _train_loss, _train_acc = self._train()
            _test_loss, _test_acc = self._test()

            # plot loss, plot acc
            train_loss.append(_train_loss)
            test_loss.append(_test_loss)
            train_acc.append(_train_acc)
            test_acc.append(_test_acc)
            self._loss_plot(epoch, train_loss, test_loss)
            self._acc_plot(epoch, train_acc, test_acc)

            # save model
            if epoch % 10 == 0:
                print('>>> save {0:04d}.model'.format(epoch))
                serializers.save_hdf5(self.save_dir + '/{0:04d}.model'.format(epoch), self.model)
                serializers.save_hdf5(self.save_dir + '/{0:04d}.state'.format(epoch), self.optimizer)

    def _random_batches(self, data) -> list:
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

    def _forward(self, feature_batch: np.ndarray, label_batch: np.ndarray, train=True):
        # 0軸 と 1軸を入れ替えて転置
        _x = feature_batch.transpose((1, 0, 2))
        # gpu を使っていれば、cupy に変換される
        x = self.xp.asarray(_x).astype(np.float32)
        t = self.xp.asarray(label_batch).astype(np.int32)
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)
                # if not train:
                #     print('==== softmax ==========================')
                #     print(F.softmax(y).data)
                #     print('==== label ==========================')
                #     print(t)
        return loss, accuracy

    def _train(self):
        loss, acc = [], []
        batch_num = int(len(self.train_data[1])/self.BACH_SIZE)

        for i, (feature_batch, label_batch) in enumerate(self._random_batches(self.train_data)):
            self.model.cleargrads()
            _loss, _acc = self._forward(feature_batch, label_batch)
            _loss.backward()
            # _loss.unchain_backward()  # TODO: これは必要なのだろうか??
            self.optimizer.update()
            loss.append(_loss.data.tolist())
            acc.append(_acc.data.tolist())

            if i % 100 == 0:
                loop = i + 1
                print('{} / {} loss: {} accuracy: {}'.format(loop, batch_num, str(np.average(loss)), str(np.average(acc))))
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('<<< _train loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _test(self):
        loss, acc = [], []
        for i, (feature_batch, label_batch) in enumerate(self._random_batches(self.test_data)):
            _loss, _acc = self._forward(feature_batch, label_batch, train=False)
            # print('loss:', loss)
            # print('acc:', acc)
            # time.sleep(3)
            loss.append(_loss.data.tolist())
            acc.append(_acc.data.tolist())
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('<<< test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    @staticmethod
    def _loss_plot(epoch, train_loss, test_loss):
        plt.cla()
        plt.plot(np.arange(epoch+1), np.array(train_loss))
        plt.plot(np.arange(epoch+1), np.array(test_loss))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('loss.png')

    @staticmethod
    def _acc_plot(epoch, train_acc, test_acc):
        plt.cla()
        plt.plot(np.arange(epoch+1), np.array(train_acc))
        plt.plot(np.arange(epoch+1), np.array(test_acc))
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.savefig('acc.png')

    @staticmethod
    def _print_title(s: str):
        """
        現在の処理内容とデバッグしたいデータ（配列）を渡す）
        :param str s:
        :return:
        """
        print()
        print('<<<< Train class:', s, '>>>>')


if __name__ == '__main__':
    # setup
    Train.GPU_DEVICE = 0

    # params
    framed_cnn_pkl_path = './output/framed_cnn/20180929_075743.pkl'

    train = Train(framed_cnn_pkl_path)
    train.main()
