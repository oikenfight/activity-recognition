import numpy as np
import json
import os
import pickle as pkl
from FileManager import FileManager
from image_model import VGG19
import time
from datetime import datetime


class Features:
    BASE = './src/cnn/'
    INPUT_FILES_BASE = './src/frames/frame_data'
    OUTPUT_BASE = BASE + 'features/'
    OUTPUT_FILENAME = OUTPUT_BASE + 'sample.pkl'
    FileManager.BASE_DIR = INPUT_FILES_BASE
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'
    image_model = VGG19()
    file_manager = FileManager()

    def __init__(self):
        self.all_file_lists = self.file_manager.all_file_lists()
        self.features = {}
        self.load_model()

    def load_model(self):
        self.image_model.load(self.MODEL_PATH)

    def main(self):
        for path_data in self.all_file_lists:
            # パスを取得
            path = self.get_path(path_data)
            # 画像から特徴量を取得
            feature = self.get_feature(path)
            print(feature)
            # 特徴量を追加f
            self.append(path_data, feature)
            # pkl に dump
            self.dump_to_pkl()

    def get_path(self, path_data: list) -> str or None:
        path = self.INPUT_FILES_BASE + '/' + '/'.join(path_data)
        if os.path.exists(path):
            return path
        else:
            print('============================')
            print('image file not found.')
            print(path)
            exit()
            print('============================')
            return None

    def get_feature(self, path) -> np.array:
        """
        :param str path: this is model path
        :return:
        """
        return np.array(self.image_model.feature(path))

    def append(self, path_data: list, feature: np.array):
        # アクション名が辞書に登録されていなければ追加
        if path_data[0] not in self.features:
            self.features[path_data[0]] = []
        # 対応アクションにfeatureを追加
        self.features[path_data[0]].append(feature)

    def dump_to_pkl(self):
        for action in self.features.items():
            for feature in action:
                print(feature)
        # with open(self.OUTPUT_FILENAME, 'wb') as f:
        #     pkl.dump(self.features, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # set up FileManager
    BASE = './src/cnn/'
    INPUT_FILES_BASE = './src/frames/frame_data'
    OUTPUT_BASE = BASE + 'features/'
    OUTPUT_FILENAME = OUTPUT_BASE + 'sample.pkl'
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'

    # set up
    Features.BASE = BASE
    Features.BASE_OUTPUT = OUTPUT_BASE
    Features.OUTPUT_FILENAME = OUTPUT_BASE + 'test.pkl'
    Features.MODEL_PATH = MODEL_PATH
    features = Features()

    # execute
    features.main()
