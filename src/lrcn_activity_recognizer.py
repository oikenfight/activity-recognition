import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pickle as pkl
import chainer
from chainer import cuda, serializers, Variable, functions as F
from lrcn_recognition.LrcnActivityRecognitionModel import LrcnActivityRecognitionModel
import img2vec
import matplotlib
from copy import deepcopy

matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()


class LrcnActivityRecognizer:
    """
    LRCN を用いて時系列画像から行動認識を実行する。
    認識は各フレーム毎に行い、そのフレームの行動を推定する。
    """

    # constant
    MODEL_PATH = ''

    # set up
    GPU_DEVICE = 1

    def __init__(self, actions: dict):
        self.xp = np
        self.actions = actions
        self.model = None
        self.img2vec_converter_instance = img2vec.converter.Converter()
        self._prepare()

    def _prepare(self):
        self._set_model()
        self._to_gpu()

    def _set_model(self):
        print('>>> set LrcnActivityRecognitionModel')
        self.model = LrcnActivityRecognitionModel(len(self.actions))
        serializers.load_hdf5(self.MODEL_PATH, self.model)

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            print('>>> use gpu')
            self.xp = cuda.cupy
            self.model.to_gpu()

    def main(self, paths: list, label: int):
        """
        画像パスのリストとその正解ラベルを受け取り、推測結果と正解ラベルの確度を返却
        :param paths: string list of paths
        :param label: int
        :return:
        """
        img_paths = self._sanitized_inputs(paths)
        img_vecs = self._img_to_vec(img_paths)

        model = deepcopy(self.model)

        model.lstm_reset_state()
        model.cleargrads()

        # 各画像毎の結果を格納する
        ys = []

        # この中では推論処理を行う（学習時と推論時で動作が異なる場合に有効だけど、今回は特に関係ないけどね。）
        with chainer.using_config('train', False):
            # 計算後にロス関数の各パラメータについての勾配のため、内部に計算グラフを保持するかどうか。推論のときはメモリ節約のため False
            with chainer.using_config('enable_backprop', False):

                for i in range(len(img_vecs)):
                    # gpu を使っていれば、cupy に変換される
                    # バッチサイズ 1 のベクトルを渡してる（学習時と同様に処理するため）
                    x = self.xp.array([img_vecs[i]]).astype(np.float32)
                    y = F.softmax(model(x)).data[0]
                    ys.append(y)
        # メモリ解放
        model = None

        # 各画像の結果をまとめる
        ys = self.xp.array(ys, dtype=np.float32)
        result = self.xp.average(ys, axis=0)
        max_value, max_label = float(result.max()), int(result.argmax())

        self._predict_action(max_label)
        # print('max_label:', str(max_label), 'max:', str(max_value))
        return max_label, max_value, label, float(result[label])

    def _sanitized_inputs(self, inputs: list) -> list:
        """
        入力されるパスを、適切なパスに置換する（人物検出が成功したものを基準となる。構造は同じだが、ベースとなる部分が異なる。）
        TODO: 人物検出をどうするか未定だから、とりあえず何もしないで頑張る
        :param inputs: list of path string
        :return list: list of sanitized path strings
        """
        return inputs

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

    def _predict_action(self, max_index: int):
        result_action = self.actions[max_index]
        # print('>>>> result action is', result_action)


if __name__ == "__main__":
    # set up
    LrcnActivityRecognizer.MODEL_PATH = 'output/lrcn_recognition/models/20181206_202352/0140.model'
    ACTIONS_PATH = 'output/lrcn_recognition/models/20181206_202352/actions.pkl'
    # params
    paths = [
        '/images_data/20181204_013516/drinking/a001-0508C/0005.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0006.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0007.jpg',
        '/images_data/20181204_013516/drinking/a001-0508C/0008.jpg',
    ]
    with open(ACTIONS_PATH, mode="rb") as f:
        actions = pkl.load(f)
    print('action length: ', str(len(actions)))
    print(actions)
    label = 16

    # instance
    cnn_predict_instance = LrcnActivityRecognizer(actions)
    cnn_predict_instance.main(paths, label)
