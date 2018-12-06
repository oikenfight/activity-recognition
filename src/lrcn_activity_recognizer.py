import numpy as np
import pickle as pkl
import chainer
from chainer import serializers, functions as F
import cupy
from lrcn_recognition.LrcnActivityRecognitionModel import LrcnActivityRecognitionModel
import img2vec
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()


class LrcnActivityRecognizer:
    """
    LRCN を用いて時系列画像から行動認識を実行する。
    認識は各フレーム毎に行い、そのフレームの行動を推定する。
    """

    # constant
    INPUT_IMAGE_DIR = '/images_data/{datetime}/'
    MODEL_PATH = ''
    ACTIONS_PATH = ''

    # set up
    GPU_DEVICE = 0
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self):
        self.xp = np
        self.actions = None
        self.model = None
        self.img2vec_converter_instance = img2vec.converter.Converter()

    def _prepare(self):
        self._load_actions()
        self._set_model()
        self._to_gpu()

    def _load_actions(self):
        with open(self.ACTIONS_PATH, mode="rb") as f:
            self.actions = pkl.load(f)

    def _set_model(self):
        print('>>> set CnnActivityRecognitionModel')
        self.model = LrcnActivityRecognitionModel(len(self.actions))
        serializers.load_hdf5(self.MODEL_PATH, self.model)

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            print('>>> use gpu')
            self.xp = cupy
            self.model.to_gpu(self.GPU_DEVICE)

    def main(self, inputs: list, label: int):
        img_paths = self._sanitized_inputs(inputs)
        img_vecs = self._img_to_vec(img_paths)

        # テスト実行
        self.model.cleargrads()
        return self._forward(img_vecs, label)

    def _sanitized_inputs(self, inputs: list) -> list:
        """
        入力されるパスを、適切なパスに置換する（人物検出が成功したものを基準となる。構造は同じだが、ベースとなる部分が異なる。）
        :param inputs: list of path string
        :return list: list of sanitized path strings
        """
        return []

    def _img_to_vec(self, img_paths: list) -> list:
        """
        画像のパスのリストを受け、ベクトルに変換してリストに格納し返却する
        :param img_paths:
        :return img_vecs:
        """
        img_vecs = []
        for path in img_paths:
            img_vecs.append(self.img2vec_converter_instance.main(path))
        return img_vecs

    def _forward(self, img_vecs: list, label: int) -> tuple:
        """
        順方向計算実行。
        :param cupy.ndarray image_vec_batch:
        :param np.ndarray label_batch:
        :param bool train:
        :return:
        """
        x = self.xp.array([img_vecs]).astype(np.float32)
        t = self.xp.array(label).astype(np.int32)

        # gpu を使っていれば、cupy に変換される
        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)
                accuracy = F.accuracy(y, t)
        return loss, accuracy


if __name__ == "__main__":
    # set up
    LrcnActivityRecognizer.INPUT_IMAGE_DIR = '/images_data/{datetime}/'
    LrcnActivityRecognizer.MODEL_PATH = ''
    LrcnActivityRecognizer.ACTIONS_PATH = ''
    # params
    paths = [
        '/images_data/20181204_013516/drinking/a001-0508C/0001.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0002.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0003.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0004.jpg',
    ]
    label = 0
    # instance
    cnn_predict_instance = LrcnActivityRecognizer()
    cnn_predict_instance.main(paths, label)
