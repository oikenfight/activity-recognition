from chainer import cuda
import numpy as np
import cupy as cp
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from image_model import VGG19

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Cnn:
    # constant
    GPU_DEVICE = 0
    MODEL_PATH = './src/cnn/VGG_ILSVRC_19_layers.caffemodel'

    def __init__(self):
        self.image_model = self._set_image_model()
        self._to_gpu()

    def _set_image_model(self):
        self._print_title('set image_model')
        image_model = VGG19()
        image_model.load(self.MODEL_PATH)
        return image_model

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            self.image_model.to_gpu(self.GPU_DEVICE)

    def main(self, input_path: str) -> list:
        self._print_title('main method')
        if not os.path.isfile(input_path):
            raise InputFileNotFoundError
        return self._get_feature(input_path)

    def main_with_dir(self, input_dir_path) -> list:
        # self._print_title('main method when param input path is dir: %s' % input_dir_path)
        # 指定したディレクトリ以下のファイルを取得
        FileManager.BASE_DIR = input_dir_path
        file_manager = FileManager()
        all_file_list = file_manager.all_file_lists()

        # 特徴抽出
        for i, to_file_path_data in enumerate(all_file_list):
            path = self._get_path(input_dir_path, to_file_path_data)
            yield self._get_feature(path)

    @staticmethod
    def _get_path(input_dir_path: str, to_file_path_data: list) -> str or None:
        """
        ファイルの存在を確認し、パスを返却
        :param str input_dir_path, list to_file_path_data: 探索対象のディレクトリのパスと、探索対象以下のファイルへのパスリスト
        :return:
        """
        path = input_dir_path + '/'.join(to_file_path_data)
        if not os.path.isfile(path):
            raise InputFileNotFoundError
        return path

    def _get_feature(self, path) -> np.array:
        """
        特徴抽出実行
        :param str path: this is model path
        :return:
        """
        # list として保存しておいたほうが処理が早そうなため。（GPU 使ってないときは一個目のじゃないとエラーになるよ）
        if self.GPU_DEVICE >= 0:
            return cp.asnumpy(self.image_model.feature(path).data[0]).tolist()
        else:
            return self.image_model.feature(path).data[0]

    @staticmethod
    def _print_title(string: str):
        print()
        print("<<< Cnn Class: %s >>>" % string)


class InputFileNotFoundError(Exception):
    def __str__(self):
        return "入力されたファイルを見つけることができませんでした。"


if __name__ == "__main__":
    #
    # example
    #

    # params
    test_file_path = './tmp/0001.jpg'
    test_dir_path = './tmp/'

    # set up
    Cnn.MODEL_PATH = './src/cnn/VGG_ILSVRC_19_layers.caffemodel'
    Cnn.GPU_DEVICE = 0

    #
    # param is file path
    #

    # set up
    cnn_instance = Cnn()

    # execute
    cnn_vec = cnn_instance.main(test_file_path)
    print(len(cnn_vec))

    # execute with directory
    for i, cnn_vec in enumerate(cnn_instance.main_with_dir(test_dir_path)):
        print(str(i), ': ', str(len(cnn_vec)))
