import numpy as np
import pickle as pkl
from datetime import datetime
import cupy as cp
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from cnn import Cnn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class CnnBatch:
    # constant
    BASE = './src/cnn/'

    # setup
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'
    GPU_DEVICE = 0

    def __init__(self, input_base_dir: str, output_base_dir: str):
        self.input_base_dir = input_base_dir
        self.output_dir = self._set_output_dir(output_base_dir)
        self.file_manager = None
        self.features = {}

    @staticmethod
    def _set_output_dir(output_dir_base: str) -> str:
        return output_dir_base + datetime.now().strftime("%Y%m%d_%H%M%S") + '/'

    def _prepare(self):
        self._print_title('prepare')

        # ファイルを取得する準備
        FileManager.BASE_DIR = self.input_base_dir
        self.file_manager = FileManager()
        # 保存先ディレクトリを作成
        os.makedirs(self.output_dir)

    def main(self):
        self._print_title('main method')
        self._prepare()

        # setup
        Cnn.MODEL_PATH = self.MODEL_PATH
        Cnn.GPU_DEVICE = self.GPU_DEVICE
        cnn_instance = Cnn()

        input_dir_path_list = self._get_input_dir_path_list()
        for k, (action_name, video_list) in enumerate(input_dir_path_list.items()):
            print('>>> (%s / %s): %s' % (str(k), str(len(input_dir_path_list)), action_name))
            for i, video_name in enumerate(video_list):
                # params
                input_dir_path = self.input_base_dir + action_name + '/' + video_name + '/'
                # execute
                for cnn_vec in cnn_instance.main_with_dir(input_dir_path):
                    if video_name not in self.features: self.features[video_name] = []
                    self.features[video_name].append(cnn_vec)
                if i % 100 == 0:
                    print('%s / %s completed' % (str(i), str(len(video_list))))

            # dump features to ...
            dump_path = self.output_dir + action_name + '.pkl'
            self._dump_to(dump_path)
            # 次のアクションに備えて一旦初期化
            self.features = {}

    def _get_input_dir_path_list(self) -> dict:
        # ディレクトリ辞書を生成
        input_dir_path_list = {}
        for path_data in self.file_manager.all_file_lists():
            """
            path_data[0]: アクション名
            path_data[1]: ビデオファイル名
            path_data[2]: 画像名 => これは完全に無視
            """
            if path_data[0] not in input_dir_path_list:
                # 新しいアクションをキーに、空リストデータを作成
                input_dir_path_list[path_data[0]] = []
            if path_data[1] not in input_dir_path_list[path_data[0]]:
                # アクション名をキーに、ビデオファイル名が存在しなければデータを追加
                input_dir_path_list[path_data[0]].append(path_data[1])
        return input_dir_path_list

    def _dump_to(self, path):
        self._print_title('dump to %s' % path)
        print('dump to: ' + path)
        print('data size:', str(len(self.features)))
        with open(path, 'wb') as f:
            pkl.dump(self.features, f, pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _print_title(string: str):
        print()
        print("<<< CnnBatch Class: %s >>>" % string)


if __name__ == "__main__":
    #
    # Example
    #

    # setup
    BASE = './src/with_trained_model/cnn/'
    MODEL_PATH = BASE + 'VGG_ILSVRC_19_layers.caffemodel'
    GPU_DEVICE = 0

    CnnBatch.BASE = BASE
    CnnBatch.MODEL_PATH = MODEL_PATH
    CnnBatch.BASE = GPU_DEVICE

    # params
    input_base_dir = '/converted_data/20180929_071816/'
    output_base_dir = './output/cnn/'

    # execute
    cnn_batch_instance = CnnBatch(input_base_dir, output_base_dir)
    cnn_batch_instance.main()


