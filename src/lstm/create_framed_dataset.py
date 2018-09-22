import pickle as pkl
import numpy as np
from chainer.datasets import tuple_dataset
from FileManager import FileManager
from chainer import cuda
import time
from datetime import datetime


class Dataset:
    # TASK: 要セットアップ
    BASE = '.'
    INPUT_FILES_BASE = '.'
    OUTPUT_FILE_DIR = './output/lstm_frame/'
    FEATURE_SIZE = 4096
    GPU_ID = 0
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self):
        self.to_gpu()
        self.all_files = self.load_input_files()  # List(str)
        self.actions = {}
        # list のが処理が速いらしいから、あえて numpy じゃなくて list にしてみた。
        # self.label_data = np.empty(0, int)
        # self.features_data = np.empty((0, self.FRAME_SIZE, self.FEATURE_SIZE), float)
        self.features_data = []
        self.label_data = []
        self.load()

    def to_gpu(self):
        print('use gup')
        cuda.check_cuda_available()
        cuda.get_device(self.GPU_ID).use()

    def load_input_files(self):
        FileManager.BASE_DIR = self.INPUT_FILES_BASE
        file_manager = FileManager()
        return file_manager.all_files()

    def load(self):
        print('Dataset Loading ...')
        print('action length: ', str(len(self.all_files)))
        for i, pkl_file in enumerate(self.all_files):
            print('=================================')
            print(pkl_file)

            # actions 辞書作成（どのラベルがどのアクションなのか後で分かるようにするため。pkl_file の中身次第だから、よしなに。）
            action_name = pkl_file.split('.')[1].split('/')[-1]
            self.actions[len(self.actions)] = action_name

            # # 正解ラベル作成
            # action_label = np.zeros(len(self.all_files))
            # action_label[i] = 1
            # print(action_label)

            with open(pkl_file, 'rb') as f:
                dataset = pkl.load(f)
            print('data number: ', str(len(dataset)))

            # ラベルと時系列画像特徴量リストが対応するデータセットを作成
            for k, (folder_name, features) in enumerate(dataset.items()):
                for n in range(0, len(features), self.OVERLAP_SIZE):
                    frame_data = features[n:n+self.FRAME_SIZE]
                    # TODO: 微妙に 8 フレームに足りてないデータを無駄にしてるから、できたら直す（学習する時にフレーム数が一定のほうが楽だから今はこうしてる。）
                    if len(frame_data) < 8: break
                    # self.label_data = np.append(self.label_data, np.array(i))
                    # self.features_data = np.append(self.features_data, np.array([frame_data]), axis=0)
                    self.label_data += [i]
                    self.features_data += [[frame_data]]

                if k % 50 == 0:
                    # print(str(k), '/', len(dataset), ':', folder_name, ', data:', self.features_data.shape, ', label:', self.label_data.shape)
                    print(str(k), '/', len(dataset), ':', folder_name, ', data:(', len(self.features_data), len(self.features_data[0]), len(self.features_data[0][0]), '), label:', len(self.label_data))

    def _output_file_name(self):
        return self.OUTPUT_FILE_DIR + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl'

    def dump_to_framed_pkl(self):
        print('dumping to ...')
        print(len(self.features_data))

        data = {
            'label': self.label_data,
            'features': self.features_data,
            'actions_dict': self.actions,
        }

        with open(self._output_file_name(), 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # set up
    Dataset.BASE = './src/lstm/'
    # Dataset.INPUT_FILES_BASE = './output/cnn/20180908_051340'
    Dataset.INPUT_FILES_BASE = './output/cnn/20180913_083117'
    OUTPUT_FILE_DIR = './output/lstm_frame/'
    dataset = Dataset()
    dataset.dump_to_framed_pkl()
