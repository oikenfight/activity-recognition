import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pickle as pkl
import chainer
from chainer import cuda, datasets, iterators, optimizers, serializers, functions as F
from datetime import datetime
import os
from glob import glob
import matplotlib
import threading
from FileManager import FileManager
from lrcn_recognition.LrcnActivityRecognitionModel import LrcnActivityRecognitionModel
import img2vec
from pprint import pprint as pp
import random

from multiprocessing import Pool

matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()


class Train:
    """
    LRCN を用いて、時系列画像から行動認識してみる。
    全画像データを読み込むのがメモリの問題で不可能だったため、
    一気に全画像を読み込むのではなく、可能な限り多くの画像をランダムに読み込んでおき、
    適当なタイミングでデータをシャッフルしてまた読み込んで学習する。
    """

    # Train constant
    INPUT_IMAGE_DIR = '/converted_data/{datetime}/'
    OUTPUT_BASE = './output/lrcn_recognition/models/'

    # setup
    GPU_DEVICE = 0
    EPOCH_NUM = 100000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 85
    TEST_RATE = 0.4
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4
    KEPT_FRAME_SIZE = 12000
    KEPT_TEST_FRAME_SIZE = 3000
    THREAD_SIZE = 100
    TRAIN_RATE = 0.05

    def __init__(self):
        self.xp = np
        self.actions = {}  # ラベル:アクション名 索引辞書
        self.frames = np.array([], dtype=str)  # フレームに区切られたビデオ（時系列画像）のパスデータを格納（labels, images と順同一）
        self.labels = np.array([], dtype=np.int8)  # 各 video の正解ラベルを格納（frames, images と順同一）
        self.train_indexes = np.array([], dtype=np.int32)
        self.test_indexes = np.array([], dtype=np.int32)

        # chainer 由来のやつら
        self.model, self.optimizer, self.save_dir = None, None, None
        # # gpu で並列処理させるために使うやつ
        # self.comm = chainermn.create_communicator('hierarchical')

        # その他、フラグ
        self.save_dir = ''
        self.training = True
        self.log_file = None
        self.loaded_num = 0

        # インスタンスを準備
        self.img2vec_converter_instance = img2vec.converter.Converter()

        # 学習に使用するデータを常にランダムに更新しながら保持（全データはメモリの都合上読み込めないため。）
        # indexes, frames, labels, images_vec は順同一
        self.kept_indexes = np.empty(self.KEPT_FRAME_SIZE, dtype=np.int32)   # 現在学習に用いているなデータの index を格納
        self.kept_frames = np.empty((self.KEPT_FRAME_SIZE, self.FRAME_SIZE), dtype='S96')  # 現在学習に用いている各フレームの時系列画像のパスデータ(str array)を格納
        self.kept_labels = np.empty(self.KEPT_FRAME_SIZE, dtype=np.int8)    # kept_frames に順対応した正解ラベルを格納
        self.kept_images_vec = np.empty((self.KEPT_FRAME_SIZE, self.FRAME_SIZE, self.IMAGE_CHANNEL, self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.float32)  # kept_frames に順対応したベクトルデータを格納

        # 学習と同様にテストデータも保持する
        self.kept_test_indexes = np.empty(self.KEPT_TEST_FRAME_SIZE, dtype=np.int32)
        self.kept_test_frames = np.empty((self.KEPT_TEST_FRAME_SIZE, self.FRAME_SIZE), dtype='S96')
        self.kept_test_labels = np.empty(self.KEPT_TEST_FRAME_SIZE, dtype=np.int8)
        self.kept_test_images_vec = np.empty((self.KEPT_TEST_FRAME_SIZE, self.FRAME_SIZE, self.IMAGE_CHANNEL, self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.float32)

    def _prepare(self):
        self._load_frames_and_labels()
        self._shuffle_data()
        self._set_model_and_optimizer()
        self._make_save_dir()
        self._dump_actions_data()
        self._dump_test_data()
        self._dump_train_data()
        self._to_gpu()
        self._init_kept_data()

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            self.xp = cuda.cupy
            self.model.to_gpu()
            print('>>> use gpu')

    def _load_frames_and_labels(self):
        """
        画像
        :return void:
        """
        print('>>> load sentences')
        FileManager.BASE_DIR = self.INPUT_IMAGE_DIR
        file_manager = FileManager()

        # setup
        current_action = ''
        current_label = 0
        frames = []
        labels = []

        for path_list in file_manager.all_dir_lists():
            """
            path_list[0]: アクション名
            path_list[1]: ビデオ名
            """
            # 必要なパスを取得
            path = '/'.join(path_list)
            # アクションが更新されたら状態を変更する
            if not path_list[0] == current_action:
                current_action = path_list[0]
                current_label = len(self.actions)
                self.actions[len(self.actions)] = path_list[0]

            # ビデオ内のファイル一覧を取得
            files = sorted(glob(self.INPUT_IMAGE_DIR + path + "/*.jpg"))

            # フレームを作成して、ラベルと共に追加
            for i in range(0, len(files), self.OVERLAP_SIZE):
                frame = files[i:i + self.FRAME_SIZE]
                # TODO: 微妙に 8 フレームに足りてないデータを無駄にしてるから、できたら直す（学習する時にフレーム数が一定のほうが楽だから今はこうしてる。）
                if len(frame) < 8:
                    break
                # 追加
                frames.append(frame)
                labels.append(current_label)

            # if len(frames) > 10:
            #     break

        # インスタンス変数に値をセット
        self.frames = np.array(frames)
        self.labels = np.array(labels)

    def _shuffle_data(self):
        """
        任意の割合でランダムに訓練データとテストデータの tuple を返却する
        :return void:
        """
        print('>>> shuffle data to train and test')

        # index（データ数配列）をランダムに並び替え、2つにデータを分ける
        test_num = int(len(self.labels) * self.TEST_RATE)
        indexes = np.array(range(len(self.labels)))
        np.random.shuffle(indexes)

        # シャッフルされた index を train, test にそれぞれ振り分け格納
        self.train_indexes = indexes[test_num:]
        self.test_indexes = indexes[:test_num]

    def _set_model_and_optimizer(self) -> LrcnActivityRecognitionModel:
        self._print_title('set model:')
        self.model = LrcnActivityRecognitionModel(len(self.actions))

        # # TODO: 途中まで学習したモデルを使う（時間省略のため暫定的）
        # tmp_model_path = './output/lrcn_recognition/models/20190101_063332/0059.model'
        # serializers.load_hdf5(tmp_model_path, self.model)

        optimizer = chainer.optimizers.MomentumSGD()
        self.optimizer = optimizer.setup(self.model)

        # # GPU デバイスを複数使う為に変更
        # optimizer = chainermn.create_multi_node_optimizer(
        #     chainer.optimizers.MomentumSGD(), self.comm)
        # self.optimizer = optimizer.setup(self.model)

        # 学習済みレイヤの学習率を抑制
        for func_name in self.model.base._children:
            for param in self.model.base[func_name].params():
                param.update_rule.hyperparam.lr *= self.TRAIN_RATE
        # # 学習済みレイヤを完全に固定
        # self.model.base.disable_update()

    def _make_save_dir(self):
        save_dir = self.OUTPUT_BASE + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir)
        self.save_dir = save_dir

    def _dump_actions_data(self):
        print('dump to actions into : ' + self.save_dir)
        path = self.save_dir + '/actions.pkl'

        with open(path, 'wb') as f:
            pkl.dump(self.actions, f, pkl.HIGHEST_PROTOCOL)

    def _dump_test_data(self):
        """
        精度検証時にテストに用いたデータのみで検証を行いたいため、追加したメソッド
        test_indexes を用いて、テストデータのパス一覧を保存する
        :return:
        """
        output_data = []
        for frame in self.frames[self.test_indexes]:
            output_data.append(frame)

        path = self.save_dir + '/test_frame_data.pkl'
        print('dump test frame data : ', path)

        with open(path, 'wb') as f:
            pkl.dump(output_data, f, pkl.HIGHEST_PROTOCOL)

    def _dump_train_data(self):
        """
        精度検証時に学習に用いたデータのみで検証を行いたいため、追加したメソッド
        test_indexes を用いて、テストデータのパス一覧を保存する
        :return:
        """
        output_data = []
        for frame in self.frames[self.train_indexes]:
            output_data.append(frame)

        path = self.save_dir + '/train_frame_data.pkl'
        print('dump train frame data : ', path)

        with open(path, 'wb') as f:
            pkl.dump(output_data, f, pkl.HIGHEST_PROTOCOL)

    def _init_kept_data(self):
        self.loaded_num = 0

        # 学習に使用するデータを常にランダムに更新しながら保持（全データはメモリの都合上読み込めないため。）
        # indexes, frames, labels, images_vec は順同一
        self.kept_indexes = np.empty(self.KEPT_FRAME_SIZE, dtype=np.int32)   # 現在学習に用いているなデータの index を格納
        self.kept_frames = np.empty((self.KEPT_FRAME_SIZE, self.FRAME_SIZE), dtype='S96')  # 現在学習に用いている各フレームの時系列画像のパスデータ(str array)を格納
        self.kept_labels = np.empty(self.KEPT_FRAME_SIZE, dtype=np.int8)    # kept_frames に順対応した正解ラベルを格納
        self.kept_images_vec = np.empty((self.KEPT_FRAME_SIZE, self.FRAME_SIZE, self.IMAGE_CHANNEL, self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.float32)  # kept_frames に順対応したベクトルデータを格納

        # 学習と同様にテストデータも保持する
        self.kept_test_indexes = np.empty(self.KEPT_TEST_FRAME_SIZE, dtype=np.int32)
        self.kept_test_frames = np.empty((self.KEPT_TEST_FRAME_SIZE, self.FRAME_SIZE), dtype='S96')
        self.kept_test_labels = np.empty(self.KEPT_TEST_FRAME_SIZE, dtype=np.int8)
        self.kept_test_images_vec = np.empty((self.KEPT_TEST_FRAME_SIZE, self.FRAME_SIZE, self.IMAGE_CHANNEL, self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.float32)

        self._set_kept_data()
        self._set_kept_test_data()

    def _set_kept_data(self):
        """
        学習のために保持する画像の初期データを読み込む。
        :return:
        """
        print('>>> _set_init_kept_images')
        np.random.shuffle(self.train_indexes)
        init_kept_indexes = self.train_indexes[:self.KEPT_FRAME_SIZE]

        all_target_indexes = [init_kept_indexes[i:i+self.THREAD_SIZE] for i in range(0, self.KEPT_FRAME_SIZE, self.THREAD_SIZE)]
        # p = Pool(8)
        # images_data_by_thread = p.map(_load_images_from_frame_indexes, all_target_indexes)
        # p.close()

        current_index = 0
        for n in range(0, len(all_target_indexes), 8):
            p = Pool(8)
            images_data_by_thread = p.map(_load_images_from_frame_indexes, all_target_indexes[n:n+8])
            p.close()

            for i, images_data in enumerate(images_data_by_thread):
                for j in range(len(images_data)):
                    self.kept_indexes[current_index] = images_data[j][0]
                    self.kept_frames[current_index] = np.array(images_data[j][1], dtype=str)
                    self.kept_images_vec[current_index] = np.array(images_data[j][2], dtype=np.float32)
                    self.kept_labels[current_index] = images_data[j][3]
                    current_index += 1
                    images_data[j] = ()
            images_data_by_thread = []

        # for images_data in images_data_by_thread:
        #     for j in range(len(images_data)):
        #         self.kept_indexes[current_index] = images_data[j][0]
        #         self.kept_frames[current_index] = np.array(images_data[j][1], dtype=str)
        #         self.kept_images_vec[current_index] = np.array(images_data[j][2], dtype=np.float32)
        #         self.kept_labels[current_index] = images_data[j][3]
        #         current_index += 1
        #         images_data[j] = None

    def _set_kept_test_data(self):
        """
        テストのために保持する画像の初期データを読み込む。
        （メソッド被るけど、一回しか呼ばんし面倒いので。）
        :return:
        """
        print('>>> _set_init_kept_test_images')
        np.random.shuffle(self.train_indexes)
        init_kept_test_indexes = self.test_indexes[:self.KEPT_TEST_FRAME_SIZE]

        all_target_indexes = [init_kept_test_indexes[i:i+self.THREAD_SIZE] for i in range(0, self.KEPT_TEST_FRAME_SIZE, self.THREAD_SIZE)]

        # p = Pool(8)
        # images_data_by_thread = p.map(_load_images_from_frame_indexes, all_target_indexes)
        # p.close()

        current_index = 0
        for n in range(0, len(all_target_indexes), 8):
            p = Pool(8)
            images_data_by_thread = p.map(_load_images_from_frame_indexes, all_target_indexes[n:n+8])
            p.close()

            for i, images_data in enumerate(images_data_by_thread):
                for j in range(len(images_data)):
                    self.kept_test_indexes[current_index] = images_data[j][0]
                    self.kept_test_frames[current_index] = np.array(images_data[j][1], dtype=str)
                    self.kept_test_images_vec[current_index] = np.array(images_data[j][2], dtype=np.float32)
                    self.kept_test_labels[current_index] = images_data[j][3]
                    current_index += 1
                    images_data[j] = ()
            images_data_by_thread = None

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
            self._open_log_file()  # open log file to update log
            self._print_title('epoch: {}'.format(epoch + 1))
            self._write_line_to_log('epoch: {}'.format(epoch + 1))
            self._close_log_file()  # save and close log file
            _train_loss, _train_acc = self._train()
            _test_loss, _test_acc = self._test()

            # plot loss, plot acc
            train_loss.append(_train_loss)
            train_acc.append(_train_acc)
            test_loss.append(_test_loss)
            test_acc.append(_test_acc)
            self._loss_plot(epoch, train_loss, test_loss)
            self._acc_plot(epoch, train_acc, test_acc)

            if epoch % 10 == 0 and not epoch == 0:
                self._init_kept_data()

            # save model
            if (epoch + 1) % 10 == 0 and not epoch == 1:
                print('>>> save {0:04d}.model'.format(epoch))
                self._open_log_file()  # open log file to update log
                self._write_line_to_log('>>> save {0:04d}.model'.format(epoch))
                self._close_log_file()  # save and close log file
                serializers.save_hdf5(self.save_dir + '/{0:04d}.model'.format(epoch), self.model)
                serializers.save_hdf5(self.save_dir + '/{0:04d}.state'.format(epoch), self.optimizer)

    def _train(self):
        print('>>> train start')
        self.training = True
        loss, acc = [], []
        batch_num = int(self.KEPT_FRAME_SIZE / self.BACH_SIZE)
        for i, (images_vec_batch, label_batch) in enumerate(self._random_batches_for_train()):
            # 学習実行
            self.model.lstm_reset_state()
            _loss, _acc = self._forward(images_vec_batch, label_batch)
            _loss.backward()
            self.optimizer.update()
            loss.append(_loss.data.tolist())
            acc.append(_acc.data.tolist())
            if i % 10 == 0:
                _loss_and_acc = '{} / {} loss: {} accuracy: {}'.format(i, batch_num, str(np.average(loss)), str(np.average(acc)))
                print(_loss_and_acc)
                self._open_log_file()
                self._write_line_to_log(_loss_and_acc)
                self._close_log_file()
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch train loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        self._open_log_file()
        self._write_line_to_log('train loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        self._close_log_file()
        return loss_ave, acc_ave

    def _test(self):
        print('>>> test start')
        self.training = False
        loss, acc = [], []
        batch_num = int(self.KEPT_TEST_FRAME_SIZE / self.BACH_SIZE)
        for i, (images_vec_batch, label_batch) in enumerate(self._random_batches_for_test()):
            # テスト実行
            self.model.lstm_reset_state()
            self.model.cleargrads()
            _loss, _acc = self._forward(images_vec_batch, label_batch)
            loss.append(_loss.data.tolist())
            acc.append(_acc.data.tolist())
            if i % 10 == 0:
                print('{} / {} loss: {} accuracy: {}'.format(i, batch_num, str(np.average(loss)), str(np.average(acc))))

        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        self._open_log_file()
        self._write_line_to_log('test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        self._close_log_file()
        return loss_ave, acc_ave

    def _random_batches_for_train(self) -> list:
        """
        indexes をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルの tuple 順次を返却する
        :return:
        """
        indexes = np.array(range(self.KEPT_FRAME_SIZE))
        np.random.shuffle(indexes)
        # self._check_data_consistency(indexes)
        for i in range(0, len(indexes), self.BACH_SIZE):
            batch_indexes = indexes[i:i + self.BACH_SIZE]
            yield self.kept_images_vec[batch_indexes], self.kept_labels[batch_indexes]

    def _random_batches_for_test(self) -> list:
        """
        indexes をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルの tuple 順次を返却する
        （train と共通の処理だけど変数置き換えとかするとメモリ食うから。）
        :return:
        """
        indexes = np.array(range(self.KEPT_TEST_FRAME_SIZE))
        np.random.shuffle(indexes)
        # self._check_data_consistency(indexes)
        for i in range(0, len(indexes), self.BACH_SIZE):
            batch_indexes = indexes[i:i + self.BACH_SIZE]
            yield self.kept_test_images_vec[batch_indexes], self.kept_test_labels[batch_indexes]

    def _forward(self, images_vec_batch: np.ndarray, label_batch: np.ndarray, train=True):
        """
        順方向計算実行。
        :param np.ndarray images_vec_batch:
        :param np.ndarray label_batch:
        :param bool train:
        :return:
        """
        loss = 0
        acc = 0
        self.model.lstm_reset_state()
        for i in range(self.FRAME_SIZE):
            x = self.xp.array(images_vec_batch[:, i]).astype(np.float32)
            t = self.xp.array(label_batch).astype(np.int32)
            self.model.cleargrads()

            with chainer.using_config('train', train):
                with chainer.using_config('enable_backprop', train):
                    y = self.model(x)
                    loss += F.softmax_cross_entropy(y, t)
                    acc += F.accuracy(y, t)
        return loss / float(self.FRAME_SIZE), acc / float(self.FRAME_SIZE)

    def _check_data_consistency(self, all_batch_indexes):
        acc = 0
        error = 0
        for frame, label in zip(self.kept_frames[all_batch_indexes], self.kept_labels[all_batch_indexes]):
            frame_action = frame[0].decode('utf8').split('/')[3]
            label_action = self.actions[label]
            if frame_action == label_action:
                acc += 1
            else:
                error += 1
        print('... _check_data_consistency:  acc: %s / %s, error: %s / %s' % (acc, len(all_batch_indexes), error, len(all_batch_indexes)))

    @staticmethod
    def _loss_plot(epoch, train_loss, test_loss):
        plt.cla()
        plt.plot(np.arange(epoch + 1), np.array(train_loss))
        plt.plot(np.arange(epoch + 1), np.array(test_loss))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('./output/person_lrcn_recognition/loss.png')

    @staticmethod
    def _acc_plot(epoch, train_acc, test_acc):
        plt.cla()
        plt.plot(np.arange(epoch + 1), np.array(train_acc))
        plt.plot(np.arange(epoch + 1), np.array(test_acc))
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.savefig('./output/person_lrcn_recognition/acc.png')

    def _open_log_file(self):
        path = self.save_dir + '/train.log'
        self.log_file = open(path, mode='a')

    def _write_line_to_log(self, s: str):
        self.log_file.write(s + '\n')

    def _close_log_file(self):
        self.log_file.close()

    @staticmethod
    def _print_title(s: str):
        """
        現在の処理内容とデバッグしたいデータ（配列）を渡す）
        :param str s:
        :return:
        """
        print()
        print('<<<< Train class:', s, '>>>>')

    def get_frames(self):
        return self.frames

    def get_labels(self):
        return self.labels


def _load_images_from_frame_indexes(target_indexes) -> list:
    """
    対象となるデータの index を受け、子スレッドを実行する。
    images_data: list of tuple ([images_path], [images_vec], 'label') で返却することで、
    受け取る側では、データの格納と同時に images_data[i] = None を代入してメモリ削減させる
    :param list target_indexes:
    :return:
    """
    global train
    frames = train.get_frames()
    labels = train.get_labels()

    target_frames = frames[target_indexes]
    target_labels = labels[target_indexes]
    images_data = []
    thread_list = []
    # スレッドをセット
    for i, (target_index, frame, label) in enumerate(zip(target_indexes, target_frames, target_labels)):
        thread = threading.Thread(target=_load_images_thread,
                                  args=([target_index, frame, label, images_data]))
        thread_list.append(thread)
    # スレッド実行
    for thread in thread_list:
        thread.start()
    # 全てのスレッドが完了するのを待機
    for thread in thread_list:
        thread.join()
    return images_data


def _load_images_thread(target_index: int, frame: list, label: int, images_data: list):
    """
    _update_kept_data_constantly_thread の子スレッドとして呼ばれ、与えられた frame の画像を読み込む。
    マルチスレッドとして働く。
    :param frame:
    :return:
    """
    global train

    # 画像読み込み
    images = []
    images_vec = []
    mirroring = random.choice([True, False])

    for path in frame:
        vec = train.img2vec_converter_instance.main(path, mirroring)
        images.append(path)
        images_vec.append(vec)
    # バッチデータに結果を追加
    images_data.append((target_index, frame, images_vec, label))


if __name__ == '__main__':
    # setup
    Train.GPU_DEVICE = 0
    # Train.INPUT_IMAGE_DIR = '/resized_images_data/background_pasted_images/'
    # Train.OUTPUT_BASE = './output/lrcn_recognition/models/'
    Train.INPUT_IMAGE_DIR = '/person_images_data/20190116_043313/'
    Train.OUTPUT_BASE = './output/person_lrcn_recognition/models/'
    # params
    # execute
    train = Train()
    train.main()

