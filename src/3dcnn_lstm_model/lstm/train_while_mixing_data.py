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
    """
    全画像データを読み込むのがメモリの問題で不可能だったため、
    一気に全画像を読み込むのではなく、可能な限り多くの画像をランダムに読み込んでおき、
    常にランダムに保持する画像を更新しながら、エポック数を増やすことで全画像を学習できるようにする。
    画像の更新は学習プロセスとは別に、別スレッドで更新を続ける。
    また、これをやるのは学習のときのみで、
    テスト時には、計算する度に画像を読み込むことにする。（実装が難しくなりすぎたので、時間かかるけど。。）
    """

    # Train constant
    INPUT_IMAGE_DIR = '/converted_data/{datetime}/'
    OUTPUT_BASE = './output/3dcnn_lstm_model/models/'

    # setup
    GPU_DEVICE = 0
    EPOCH_NUM = 100000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 128
    TEST_RATE = 0.2
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 320
    IMAGE_COLOR = 3
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4
    # 学習に使用するデータ数は KEPT_FRAME_SIZE だが、保持するデータ数は KEPT_FRAME_SIZE + THREAD_SIZE となる
    # （学習時に index からデータを拾う際に、kept_data が更新中で IndexError となる可能性を防ぐため。）
    KEPT_FRAME_SIZE = 3200
    THREAD_SIZE = 20

    def __init__(self):
        self.xp = np
        self.actions = {}  # ラベル:アクション名 索引辞書
        self.frames = np.array([])  # フレームに区切られたビデオ（時系列画像）のパスデータを格納（labels, images と順同一）
        self.labels = np.array([])  # 各 video の正解ラベルを格納（frames, images と順同一）
        self.train_indexes = np.array([]).astype(np.int32)
        self.test_indexes = np.array([]).astype(np.int32)
        self.model, self.optimizer, self.save_dir = None, None, None
        self.save_dir = ''
        self.update_running = True

        # 学習に使用するデータを常にランダムに更新しながら保持（全データはメモリの都合上読み込めないため。）
        # frames, labels は順同一
        self.kept_frames = np.chararray((0, 8))    # 現在学習可能な各フレームの時系列画像のパスデータ(str array)を格納
        self.kept_labels = np.empty(0).astype(np.int32)         # kept_frames に順対応した正解ラベルを格納
        # images, images_vec は順同一
        self.kept_images = np.chararray(self.KEPT_FRAME_SIZE * self.FRAME_SIZE)  # 画像パスデータを格納(str)
        self.kept_images_vec = np.empty((self.KEPT_FRAME_SIZE * self.FRAME_SIZE, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_COLOR)).astype(np.float32) # kept_images に順対応したベクトルデータを格納

    def _prepare(self):
        self._load_frames_and_labels()
        self._shuffle_data()
        self._set_model()
        self._set_optimizer()
        self._make_save_dir()
        self._dump_actions_data()
        self._to_gpu()
        self._set_init_keeping_images()
        # スレッドでランダムに常時データを更新する
        thread = threading.Thread(target=self._update_keeping_images_constantly_thread)
        thread.start()

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            self.xp = cupy
            self.model.to_gpu(self.GPU_DEVICE)
            print('>>> use gpu')

    def _load_frames_and_labels(self):
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

    def _set_init_keeping_images(self):
        """
        学習のために保持する画像の初期データを読み込む。
        :return:
        """
        print('>>> _set_init_keeping_images')
        # indexes = np.array(range(len(self.labels)))
        np.random.shuffle(self.train_indexes)
        init_kept_indexes = self.train_indexes[:self.KEPT_FRAME_SIZE]

        current_index = 0
        for i in range(0, self.KEPT_FRAME_SIZE + self.THREAD_SIZE, self.THREAD_SIZE):
            target_indexes = init_kept_indexes[i:i+self.THREAD_SIZE]
            tmp_kept_images = []
            tmp_kept_images_vec = []
            # images_data: list of tuple ([images_path], [images_vec], 'label')
            images_data = self._load_images_from_frame_indexes(target_indexes)
            for j in range(len(images_data)):
                for k in range(self.FRAME_SIZE):
                    self.kept_images[current_index] = np.array(images_data[j][0][k])
                    self.kept_images_vec[current_index] = np.array(images_data[j][1][k])
                    current_index += 1
                # tmp_kept_images += images_data[j][0]
                # tmp_kept_images_vec += images_data[j][1]
                images_data[j] = None
            # 初期画像データをセット（ここで numpy に変換するとやや遅いけど、メモリ節約のため。）
            # self.kept_images = np.append(self.kept_images, tmp_kept_images)
            # self.kept_images_vec = np.append(self.kept_images_vec, tmp_kept_images_vec, axis=0)
            print('init keeping data:', str(i+self.THREAD_SIZE), '/', str(self.KEPT_FRAME_SIZE))

        # 初期 keeping_data をセット
        self.kept_frames = self.frames[init_kept_indexes]
        self.kept_labels = self.labels[init_kept_indexes]

        print('>>>>>>> all child threads are completed.')
        print('self.kept_frames shape:', self.kept_frames.shape)
        print('self.kept_labels shape:', self.kept_labels.shape)
        print('self.kept_images shape:', self.kept_images.shape)
        print('self.kept_images_vec shape:', self.kept_images_vec.shape)

    def _update_keeping_images_constantly_thread(self):
        """
        常に動き続けるスレッドで、学習データをランダムに更新する。
        :return:
        """
        print('>>> _update_keeping_images_constantly_thread')
        while True:
            if not self.update_running:
                time.sleep(10)
            else:
                time.sleep(2)

            np.random.shuffle(self.train_indexes)
            added_target_indexes = self.train_indexes[:self.THREAD_SIZE]
            kept_images_data = self._load_images_from_frame_indexes(added_target_indexes)

            # 削除するデータを決定・取得
            random_indexes = np.array(range(self.KEPT_FRAME_SIZE))
            np.random.shuffle(random_indexes)
            removed_kept_frame_indexes = random_indexes[:self.THREAD_SIZE]
            removed_kept_label_indexes = removed_kept_frame_indexes
            removed_frames = self.frames[removed_kept_frame_indexes]

            removed_image_indexes = []
            removed_image_vec_indexes = []
            for removed_frame in removed_frames:
                """ removed_frame: str list of image_path """
                for removed_image in removed_frame:
                    """ removed_image: str """
                    # removed_image（str: 削除対象となる画像のパス）の index を探す
                    removed_target_index = np.where(self.kept_images == removed_image, 1, 0).argmax()
                    removed_image_indexes += removed_target_index
                    removed_image_vec_indexes += removed_target_index

            # 保持するフレーム・ラベルデータを更新
            self.kept_frames = np.append(self.kept_frames, self.frames[added_target_indexes], axis=0)
            self.kept_labels = np.append(self.kept_labels, self.labels[added_target_indexes], axis=0)
            self.kept_frames = np.delete(self.kept_frames, removed_kept_frame_indexes, 0)
            self.kept_labels = np.delete(self.kept_labels, removed_kept_label_indexes, 0)

            # 保持する画像データを更新
            for i, removed_image_index in enumerate(removed_image_indexes):
                self.kept_images[removed_image_index] = kept_images_data[i][0]
                self.kept_images_vec[removed_image_index] = kept_images_data[i][1]
                kept_images_data[i] = None

    def _load_images_from_frame_indexes(self, target_indexes) -> list:
        """
        対象となる frames, labels を受け、子スレッドを実行する。
        images_data: list of tuple ([images_path], [images_vec], 'label') で返却することで、
        受け取る側では、データの格納と同時に images_data[i] = None を代入してメモリ削減させる
        :param list target_frames:
        :param list target_labels:
        :return:
        """
        target_frames = self.frames[target_indexes]
        target_labels = self.labels[target_indexes]
        images_data = []
        thread_list = []
        # スレッドをセット
        for i, (frame, label) in enumerate(zip(target_frames, target_labels)):
            thread = threading.Thread(target=self._load_images_thread,
                                      args=([frame, label, images_data, i]))
            thread_list.append(thread)
        # スレッド実行
        for thread in thread_list:
            thread.start()
        # 全てのスレッドが完了するのを待機
        for thread in thread_list:
            thread.join()

        return images_data

    @staticmethod
    def _load_images_thread(frame: list, label: int, images_data: list, thread_num: int):
        """
        _update_keeping_images_constantly_thread の子スレッドとして呼ばれ、与えられた frame の画像を読み込む。
        マルチスレッドとして働く。
        :param frame:
        :return:
        """
        time1 = time.time()
        # インスタンスを準備
        img2vec_converter_instance = img2vec.converter.Converter()
        # 画像読み込み
        images = []
        images_vec = []
        for path in frame:
            # print(path)
            vec = img2vec_converter_instance.main(path)
            images.append(path)
            images_vec.append(vec)
        # バッチデータに結果を追加
        images_data.append((images, images_vec, label))
        time2 = time.time()
        # print('>>>>>> completed thread {}, time: {}'.format(str(thread_num), str(time2 - time1)))

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
            train_acc.append(_train_acc)
            test_loss.append(_test_loss)
            test_acc.append(_test_acc)
            self._loss_plot(epoch, train_loss, test_loss)
            self._acc_plot(epoch, train_acc, test_acc)

            # save model
            if epoch % 100 == 0:
                print('>>> save {0:04d}.model'.format(epoch))
                serializers.save_hdf5(self.save_dir + '/{0:04d}.model'.format(epoch), self.model)
                serializers.save_hdf5(self.save_dir + '/{0:04d}.state'.format(epoch), self.optimizer)

    def _train(self):
        print('>>> train start')
        self.update_running = True
        loss, acc = [], []
        batch_num = int(self.KEPT_FRAME_SIZE / self.BACH_SIZE)
        for i, (images_vec_batch, label_batch) in enumerate(self._random_batches_for_train(np.array(range(self.KEPT_FRAME_SIZE)))):
            # 学習実行
            print('train loop:', str(i))
            loss, acc = self._run(images_vec_batch, label_batch, loss, acc)
            print()
            if i % 20 == 0:
                print('{} / {} loss: {} accuracy: {}'.format(i, batch_num, str(np.average(loss)), str(np.average(acc))))
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch train loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _test(self):
        print('>>> test start')
        self.update_running = False
        loss, acc = [], []
        for i, (images_vec_batch, label_batch) in enumerate(self._random_batches_for_test()):
            # 学習実行
            loss, acc = self._run(images_vec_batch, label_batch, loss, acc)
        loss_ave, acc_ave = np.average(loss), np.average(acc)
        print('======= This epoch test loss: {} accuracy: {}'.format(str(loss_ave), str(acc_ave)))
        return loss_ave, acc_ave

    def _random_batches_for_train(self, indexes) -> list:
        """
        indexes をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルの tuple 順次を返却する
        :return:
        """
        np.random.shuffle(indexes)
        for i in range(0, len(indexes), self.BACH_SIZE):
            batch_indexes = indexes[i:i + self.BACH_SIZE]
            image_indexes_batch = []
            for frame in self.kept_frames[batch_indexes]:
                """ frame: str list of image_path """
                image_indexes = []
                for image in frame:
                    """ image: str """
                    # image（str: バッチ学習対象となる画像のパス）の index を取得
                    target_index = np.where(self.kept_images == image, 1, 0).argmax()
                    image_indexes += [target_index]
                image_indexes_batch += [image_indexes]
            images_vec_batch = self.xp.array([self.kept_images_vec[image_indexes] for image_indexes in image_indexes_batch])
            yield (images_vec_batch, self.kept_labels[batch_indexes])

    def _random_batches_for_test(self) -> list:
        """
        indexes をランダムな順番で、バッチサイズずつに分割したビデオ・ラベルの tuple 順次を返却する。
        テスト用の画像ベクトルデータは保持しないため、train と共通の処理ができない。。
        メモリの都合上、テストに使う画像は都度読み込ませる。
        :return:
        """
        test_batch_size = int(self.BACH_SIZE / 4)    # CPU不足すると鬼遅いから、少し小さめにした。
        test_num = int(len(self.test_indexes) / 30)
        np.random.shuffle(self.test_indexes)
        # 全テストデータの半分だけ使って、各エポックのテストは完了させる。
        for n, i in enumerate(range(0, test_num, test_batch_size)):
            print(n * test_batch_size, '/',  str(test_num))
            target_indexes = self.test_indexes[i:i + test_batch_size]
            # images_data: list of tuple ([images_path], [images_vec], 'label')
            images_data = self._load_images_from_frame_indexes(target_indexes)
            images_vec_batch = []
            label_batch = []
            for j in range(len(images_data)):
                images_vec_batch += [images_data[j][1]]
                label_batch += [images_data[j][2]]
                images_data[j] = None
            yield self.xp.array(images_vec_batch), self.xp.array(label_batch)

    def _run(self, images_vec_batch: np.ndarray, label_batch: np.ndarray, loss: list, acc: list, train=True):
        """
        計算を実行。学習時はパラメータ更新も行い、推論時は順方向計算だけ行う。
        :param images_vec_batch:
        :param label_batch:
        :param loss:
        :param acc:
        :param train:
        :return:
        """
        self.model.cleargrads()
        _loss, _acc = self._forward(images_vec_batch, label_batch)
        if train:
            _loss.backward()
            self.optimizer.update()
        loss.append(_loss.data.tolist())
        acc.append(_acc.data.tolist())
        return loss, acc

    def _forward(self, images_vec_batch: cupy.ndarray, label_batch: np.ndarray, train=True):
        """
        順方向計算実行。
        :param cupy.ndarray images_vec_batch:
        :param np.ndarray label_batch:
        :param bool train:
        :return:
        """
        # 0軸 と 1軸を入れ替えて転置
        x = images_vec_batch.astype(np.float32).transpose((1, 0, 2, 3, 4))
        t = self.xp.array(label_batch).astype(np.int32)

        # gpu を使っていれば、cupy に変換される
        with chainer.using_config('train', train):
            with chainer.using_config('enable_backprop', train):
                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)
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