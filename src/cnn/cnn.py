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
    # variable
    use_gpu = False
    image_model = VGG19()

    # constant
    GPU_DEVICE = 0
    MODEL_PATH = './src/cnn/VGG_ILSVRC_19_layers.caffemodel'

    def __init__(self, input_path):
        self.input_path = input_path    # file or directory path
        self._load_model()

        self.to_gpu()

    def _load_model(self):
        """
        モデルをロード
        :return:
        """
        self._print_title('model loading')
        self.image_model.load(self.MODEL_PATH)

    def to_gpu(self):
        self._print_title('to gpu')
        self.use_gpu = True
        self.image_model.to_gpu(self.GPU_DEVICE)

    def main(self) -> list:
        self._print_title('main method')
        if not os.path.isfile(self.input_path):
            raise InputFileNotFoundError
        return self._get_feature(self.input_path)

    def main_with_dir(self) -> list:
        self._print_title('main method when param input path is dir')
        # 指定したディレクトリ内のファイルを取得
        FileManager.BASE_DIR = self.input_path
        file_manager = FileManager()
        all_file_list = file_manager.all_file_lists()

        # 特徴抽出
        for i, path_data in enumerate(all_file_list):
            path = self._get_path(path_data)
            yield self._get_feature(path)

    def _get_path(self, path_data: list) -> str or None:
        """
        ファイルの存在を確認し、パスを返却
        :param path_data:
        :return:
        """
        path = self.input_path + '/'.join(path_data)
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
        if self.use_gpu:
            return cp.asnumpy(self.image_model.feature(path).data[0]).tolist()
        else:
            return self.image_model.feature(path).data[0]

    def _print_title(self, string: str):
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
    cnn = Cnn(test_file_path)
    # execute
    cnn_vec = cnn.main()
    print(len(cnn_vec))

    #
    # param is directory path
    #

    # set up
    cnn_with_dir = Cnn(test_dir_path)
    # execute
    for i, cnn_vec in enumerate(cnn_with_dir.main_with_dir()):
        print(str(i), ': ', str(len(cnn_vec)))
