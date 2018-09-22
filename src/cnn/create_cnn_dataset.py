import numpy as np
import pickle as pkl
from datetime import datetime
import cupy as cp
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from image_model import VGG19

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Features:
    # TASK: 要セットアップ
    BASE = './src/cnn/'
    INPUT_FILES_BASE = '/frame_data'
    OUTPUT_BASE = './output/cnn/'
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'
    image_model = VGG19()
    device = 0

    FileManager.BASE_DIR = INPUT_FILES_BASE
    file_manager = FileManager()

    def __init__(self):
        self.all_file_lists = self.file_manager.all_file_lists()
        self.features = {}
        self.save_dir = ''
        self.init()
        self.load_model()
        self.to_gpu()

    def init(self):
        """
        出力結果を保存するディレクトリを作成
        :return void:
        """
        self.save_dir = self.OUTPUT_BASE + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir)

    def load_model(self):
        """
        モデルをロード
        :return:
        """
        self.image_model.load(self.MODEL_PATH)

    def to_gpu(self):
        self.image_model.to_gpu(self.device)

    def main(self):
        all_num = len(self.all_file_lists)
        before_action = ''
        for i, path_data in enumerate(self.all_file_lists):
            # パスを取得
            path = self.get_path(path_data)
            # 画像から特徴量を取得
            feature = self.get_feature(path)
            if i % 100 == 0:
                print(str(i), '/', str(all_num), ':', path)
            # 特徴量を追加
            if before_action == path_data[0] or before_action == '':
                self.append(path_data, feature)
            else:
                # action が変わったら pkl に dump
                dump_path = self.save_dir + '/' + before_action + '.pkl'
                self.dump_to(dump_path)
                self.features = {}
                self.append(path_data, feature)
            before_action = path_data[0]

        # 一番最後のアクションを追加
        dump_path = self.save_dir + '/' + before_action + '.pkl'
        self.dump_to(dump_path)

    def get_path(self, path_data: list) -> str or None:
        """
        ファイルの存在を確認し、パスを返却
        :param path_data:
        :return:
        """
        path = self.INPUT_FILES_BASE + '/' + '/'.join(path_data)
        if os.path.exists(path):
            return path
        else:
            print('============================')
            print('image file not found.')
            print(path)
            exit()
            return None

    def get_feature(self, path) -> np.array:
        """
        特徴抽出実行
        :param str path: this is model path
        :return:
        """
        # list として保存しておいたほうが処理が早そうなため。（GPU 使ってないときは一個目のじゃないとエラーになるよ）
        # return self.image_model.feature(path).data[0]
        return cp.asnumpy(self.image_model.feature(path).data[0]).tolist()

    def append(self, path_data: list, feature: np.array):
        """
        結果を辞書に保存
        :param str path_data:
        :param np.array feature:
        :return:
        """
        if path_data[1] not in self.features:
            self.features[path_data[1]] = []
        # 対応アクションにfeatureを追加
        self.features[path_data[1]].append(feature)

    def dump_to(self, path: str):
        print('dump to: ' + path)
        print(len(self.features))

        with open(path, 'wb') as f:
            pkl.dump(self.features, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # set up FileManager
    BASE = './src/cnn/'
    INPUT_FILES_BASE = '/frame_data'
    OUTPUT_BASE = './output/cnn/'
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'

    # set up
    Features.BASE = BASE
    Features.BASE_OUTPUT = OUTPUT_BASE
    Features.INPUT_FILES_BASE = INPUT_FILES_BASE
    Features.MODEL_PATH = MODEL_PATH
    features = Features()

    # execute
    features.main()
