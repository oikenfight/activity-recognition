import numpy as np
import pickle as pkl
import chainer
from chainer import cuda, datasets, iterators, optimizers, serializers, functions as F
import time
from datetime import datetime
import os
from glob import glob
import sys
import cupy
import matplotlib
import threading

matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from ActivityRecognitionModel import ActivityRecognitionModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import img2vec
import image_preprocessor

from FileManager import FileManager


class Train:
    # Train constant
    INPUT_IMAGE_DIR = '/converted_data/{datetime}/'
    OUTPUT_BASE = './output/3dcnn_lstm_model/models/'

    # setup
    GPU_DEVICE = 0
    EPOCH_NUM = 10000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 128
    TEST_RATE = 0.25
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self):
        self.xp = np
        self.actions = {}  # ラベル:アクション名 索引辞書
        self.frames = np.array([])  # フレームに区切られたビデオ（時系列画像）のパスデータを格納（labels, images と順同一）
        self.labels = np.array([])  # 各 video の正解ラベルを格納（frames, images と順同一）
        # self.images = np.empty((0, 8, 270, 360, 3))  # 読み込んだ画像ピクセルデータを格納（frames, labels と順同一）
        self.train_indexes = np.array([])
        self.test_indexes = np.array([])
        self.model, self.optimizer, self.save_dir = None, None, None
        self.save_dir = ''

        # 学習時に使われるデータを格納
        # self.images_batch = np.empty((0, 8, 270, 360, 3))
        # self.label_batch = np.array([])
        self.images_batch = []
        self.label_batch = []

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
            print('>>> use gpu')
            self.xp = cupy
            self.model.to_gpu(self.GPU_DEVICE)

    def _load_data(self):
        """
        画像
        :return void:
        """
        print('>>> load data')
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
                self.actions[len(self.actions)] = path_list[0]
                current_action = path_list[0]
                current_label = len(self.actions)

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
            if epoch % 100 == 0:
                print('>>> save {0:04d}.model'.format(epoch))
                serializers.save_hdf5(self.save_dir + '/{0:04d}.model'.format(epoch), self.model)
                serializers.save_hdf5(self.save_dir + '/{0:04d}.state'.format(epoch), self.optimizer)

    def _train(self):
        """
        ちょっと複雑になるけど、マルチスレッド処理を使う。
        （全画像を最初に読み込んでから処理をしたかったけど、メモリが圧倒的に不足したため。）
        GPU での学習を実行する裏で、次のループの学習で使用する画像の読み込みタスクを
        を可能な限りマルチスレッドで行う（ボトルネックはここ）
        バッチサイズを極力大きくして極力効率よく学習させたい。
        画像の読み込みは、実質はフレームごとの画像ピクセル配列データが格納された pkl ファイルを読み込む感じ。
        （pillow で画像を読み込むと、速度がかなり厳しかったため、ステップ増えるけど、事前に pkl に変換しておく）
        :return:
        """
        loss, acc = [], []
        batch_num = int(len(self.train_indexes) / self.BACH_SIZE)

        for i, (next_frames_batch, next_label_batch) in enumerate(self._random_batches(self.train_indexes)):
            # TODO: ホントは下の理由で、コピーすべきだけど、コピーが重い。。から破壊的な状態（データの読み込みのが絶対遅いから、多分大丈夫。。）
            # 前回取得したデータを格納（処理速度的に可能性は低いけど、画像読み込みスレッドが先に終わった場合、インスタンス変数が書き換わるため。）
            images_batch = self.images_batch
            label_batch = self.label_batch

            # 次の処理の準備をするスレッドを作成・実行
            prepare_thread = threading.Thread(target=self._prepare_next_batch_thread, args=([next_frames_batch, next_label_batch]))
            prepare_thread.start()

            # 二回目以降は、前のループで取得したバッチを使ってスレッドで学習も行う
            if not i == 0:
                # 学習実行
                loss, acc = self._run(images_batch, label_batch, loss, acc)
                # if i % 100 == 0:
                loop = i + 1
                print('{} / {} loss: {} accuracy: {}'.format(loop, batch_num, str(np.average(loss)),
                                                             str(np.average(acc))))

            # メインスレッドを除いた実行中のスレッド一覧を取得し、次のバッチデータの読み込みが完了するのを待機
            prepare_thread.join()
            print('------ next loop ---------------')
            print()

        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch train loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _test(self):
        loss, acc = [], []

        for i, (next_frames_batch, next_label_batch) in enumerate(self._random_batches(self.test_indexes)):
            # TODO: ホントは下の理由で、コピーすべきだけど、コピーが重い。。から破壊的な状態（データの読み込みのが絶対遅いから、多分大丈夫。。）
            # 前回取得したデータを格納（処理速度的に可能性は低いけど、画像読み込みスレッドが先に終わった場合、インスタンス変数が書き換わるため。）
            images_batch = self.images_batch
            label_batch = self.label_batch

            # 次の処理の準備をするスレッドを作成・実行
            prepare_thread = threading.Thread(target=self._prepare_next_batch_thread, args=([next_frames_batch, next_label_batch]))
            prepare_thread.start()

            # 二回目以降は、前のループで取得したバッチを使ってスレッドで学習も行う
            if not i == 0:
                # 学習実行
                loss, acc = self._run(images_batch, label_batch, loss, acc, train=False)

            # メインスレッドを除いた実行中のスレッド一覧を取得し、次のバッチデータの読み込みが完了するのを待機
            prepare_thread.join()
            print('------ next loop ---------------')
            print()

        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _random_batches(self, indexes) -> list:
        """
        index をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルのリストを返却する
        :return:
        """
        np.random.shuffle(indexes)
        for i in range(0, len(indexes), self.BACH_SIZE):
            batch_indexes = indexes[i:i + self.BACH_SIZE]
            yield (self.frames[batch_indexes], self.labels[batch_indexes])

    def _prepare_next_batch_thread(self, next_frames_batch, next_label_batch):
        """
        次の学習に使用するバッチデータを受け取って、あらがじめ画像を読み込んだり、加工したり、準備するためのメソッド。
        _load_frame_and_set_data_thread メソッドをマルチスレッドで呼び出し、可能な限り分散処理させる。
        与えられたバッチデータの画像読み込みが完了したら、インスタンス変数にデータをセットすることで、処理を完了する。
        :param next_frames_batch:
        :param next_label_batch:
        """
        print('>>> _prepare_next_batch_thread')

        time1 = time.time()

        thread_list = []
        batches = []
        for i, (frame, label) in enumerate(zip(next_frames_batch, next_label_batch)):
            load_frame_thread = threading.Thread(target=self._load_frame_and_set_data_thread,
                                                 args=([frame, label, batches, i]))
            thread_list.append(load_frame_thread)

        time2 = time.time()
        print(time2 - time1)

        # スレッド実行
        for thread in thread_list:
            thread.start()

        # 全てのスレッドが完了するのを待機
        for thread in thread_list:
            thread.join()

        # TODO: 次のバッチに備え、初期化
        # TODO: スレッドが全部完了してから呼ばれるし、基本的にには学習のが速いから大丈夫なはず（ここに到達するほうが速いと破壊的になる。）
        self.images_batch = []
        self.label_batch = []

        time3 = time.time()
        print(time3 - time2)

        for i in range(len(batches)):
            self.images_batch.append(batches[i][0])
            self.label_batch.append(batches[i][1])

        print('>>>>>>> all child threads are completed.')
        print('self.images_batch length:', len(self.images_batch))
        print('self.label_batch.length:', len(self.label_batch))

    @staticmethod
    def _load_frame_and_set_data_thread(frame: list, label: int, batches: list, thread_num: int):
        """
        _prepare_next_batch_thread メソッドの子スレッドとして動くスレッド。
        frame 内の画像を読み出し、加工まで行う。
        :param frame:
        :return:
        """

        time1 = time.time()

        # インスタンスを準備
        img2vec_converter_instance = img2vec.converter.Converter()
        # 画像読み込み
        images_vec = []
        for path in frame:
            # print(path)
            vec = img2vec_converter_instance.main(path)
            images_vec.append(vec)

        # バッチデータに結果を追加
        batches.append((images_vec, label))
        time2 = time.time()
        print('>>>>>> completed thread {}, time: {}'.format(str(thread_num), str(time2 - time1)))

    def _run(self, images_batch, label_batch, loss, acc, train=True):
        """
        計算を実行。学習時はパラメータ更新も行い、推論時は順方向計算だけ行う。
        :param images_batch:
        :param label_batch:
        :param loss:
        :param acc:
        :param train:
        :return:
        """
        # print(type(images_batch))
        # print(images_batch.shape)
        self.model.cleargrads()
        _loss, _acc = self._forward(images_batch, label_batch)
        if train:
            _loss.backward()
            self.optimizer.update()
        loss.append(_loss.data.tolist())
        acc.append(_acc.data.tolist())
        return loss, acc

    def _forward(self, images_batch: np.ndarray, label_batch: np.ndarray, train=True):
        """
        順方向計算実行。
        :param images_batch:
        :param label_batch:
        :param train:
        :return:
        """
        # 0軸 と 1軸を入れ替えて転置
        x = self.xp.asarray(images_batch).astype(np.float32).transpose((1, 0, 2, 3, 4))
        t = self.xp.asarray(label_batch).astype(np.int32)

        # gpu を使っていれば、cupy に変換される
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
        plt.plot(np.arange(epoch + 1), np.array(train_loss))
        plt.plot(np.arange(epoch + 1), np.array(test_loss))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('./output/3dcnn_lstm_model/loss.png')

    @staticmethod
    def _acc_plot(epoch, train_acc, test_acc):
        plt.cla()
        plt.plot(np.arange(epoch + 1), np.array(train_acc))
        plt.plot(np.arange(epoch + 1), np.array(test_acc))
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.savefig('./output/3dcnn_lstm_model/acc.png')

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
    Train.INPUT_IMAGE_DIR = '/converted_data/20181030_120708/'
    Train.OUTPUT_BASE = './output/3dcnn_lstm_model/models/'
    # params
    # execute

    train = Train()
    train.main()
