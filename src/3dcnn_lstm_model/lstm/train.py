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
    # Train constant
    INPUT_IMAGE_DIR = '/converted_data/{datetime}/'
    OUTPUT_BASE = './output/3dcnn_lstm_model/models/'

    # setup
    GPU_DEVICE = 0
    EPOCH_NUM = 1000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 10
    TEST_RATE = 0.25
    OVERLAP_SIZE = 4

    def __init__(self):
        self.xp = np
        self.actions = {}    # ラベル:アクション名 索引辞書
        self.videos = self.xp.array([])     # video ファイルまでのパスを格納（labels と順一致）
        self.labels = self.xp.array([])     # 各 video の正解ラベルを格納（videos を順一致）
        self.train_indexes = self.xp.array([])
        self.test_indexes = self.xp.array([])
        self.model, self.optimizer, self.save_dir = None, None, None
        self.save_dir = ''

    def _prepare(self):
        self._load_data()
        self._shuffle_data()
        self._set_model()
        self._set_optimizer()
        self._make_save_dir()
        self._dump_actions_data()
        self._to_gpu()

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            self.xp = cupy
            self.model.to_gpu(self.GPU_DEVICE)
            print()
            print(type(self.xp))

    def _load_data(self):
        """
        画像
        :return void:
        """
        FileManager.BASE_DIR = self.INPUT_IMAGE_DIR
        file_manager = FileManager()

        # setup
        current_action = ''
        current_label = 0
        videos = []
        labels = []

        for path_list in file_manager.all_dir_lists():
            """
            path_list[0]: アクション名
            path_list[1]: ビデオ名
            """
            # 必要なパスを取得
            path = path_list.join('/')
            # アクションが更新されたら状態を変更する
            if not path_list[0] == current_action:
                self.actions[len(self.actions)] = path_list[0]
                current_action = path_list[0]
                current_label = len(self.actions)
            # 各データを追加
            videos.append(path)
            labels.append(current_label)
        # インスタンス変数に値をセット
        self.videos = self.xp.array(videos)
        self.labels = self.xp.array(labels)

    def _shuffle_data(self):
        """
        任意の割合でランダムに訓練データとテストデータの tuple を返却する
        :return void:
        """
        self._print_title('shuffle data to train and test')

        # index（データ数配列）をランダムに並び替え、2つにデータを分ける
        test_num = int(len(self.labels) * self.TEST_RATE)
        indexes = np.array(range(len(self.labels)))
        np.random.shuffle(indexes)

        # シャッフルされた index を train, test にそれぞれ振り分け格納
        self.train_indexes = indexes[test_num:]
        self.test_indexes = indexes[:test_num]

    def _set_model(self) -> ActivityRecognitionModel:
        # TODO: モデルを作成
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
        """
        学習を開始する
        :return:
        """
        self._print_title('main')
        self._prepare()
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        
        for epoch in range(self.EPOCH_NUM):
            self._print_title('epoch: {}'.format(epoch + 1))

            # TODO: ここから。
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
            if epoch % 30 == 0:
                print('>>> save {0:04d}.model'.format(epoch))
                serializers.save_hdf5(self.save_dir + '/{0:04d}.model'.format(epoch), self.model)
                serializers.save_hdf5(self.save_dir + '/{0:04d}.state'.format(epoch), self.optimizer)

    def _train(self):
        loss, acc = [], []
        batch_num = int(len(self.train_indexes) / self.BACH_SIZE)

        for i, (video_batch, label_batch) in enumerate(self._random_batches(self.train_indexes)):
            self.model.cleargrads()
            _loss, _acc = self._forward(video_batch, label_batch)
            _loss.backward()
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
        for i, (feature_batch, label_batch) in enumerate(self._random_batches(self.test_indexes)):
            _loss, _acc = self._forward(feature_batch, label_batch, train=False)
            loss.append(_loss.data.tolist())
            acc.append(_acc.data.tolist())
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('<<< test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _random_batches(self, indexes) -> list:
        """
        index をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルのリストを返却する
        :return:
        """
        np.random.shuffle(indexes)
        for i in range(0, len(indexes), self.BACH_SIZE):
            batch_indexes = indexes[i:i+self.BACH_SIZE]
            yield (self.videos[batch_indexes], self.labels[batch_indexes])

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
    Train.INPUT_IMAGE_DIR = '/converted_data/{datetime}/'
    Train.OUTPUT_BASE = './output/3dcnn_lstm_model/models/'
    # params
    # execute

    train = Train()
    train.main()
