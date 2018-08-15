import pickle as pkl
import numpy as np
from FileManager import FileManager

class Dataset:
    # TASK: 要セットアップ
    BASE = './src/lstm/'
    INPUT_FILES_BASE = './src/cnn/dataset/20180815_065125'
    FileManager.BASE_DIR = INPUT_FILES_BASE
    file_manager = FileManager()

    def __init__(self):
        self.all_files = self.file_manager.all_files()  # List(str)
        self.actions = {}
        self.label_data = np.array([])
        self.feature_data = np.array([])

    def load(self):
        print('Dataset Loading ...')
        print('action length: ', str(len(self.all_files)))
        for pkl_file in self.all_files:
            # actions 辞書作成
            action_label = len(self.actions)
            action_name = pkl_file.split('.')[0]
            self.actions[len(self.actions)] = action_name

            with open(pkl_file, 'rb') as f:
                dataset = pkl.load(f)

            # ラベルと時系列画像特徴量リストが対応するデータセットを作成
            for folder_name, features in dataset.items():
                np.append(self.label_data, action_label)
                np.append(self.feature_data, features)

            print('=================================')
            print(pkl_file)
            print('data number: ', str(len(dataset)))

    def get_data_of(self, index: int) -> tuple:
        return self.label_data[index], self.feature_data[index]

    def get_action_from_label(self, label_index: int) -> str:
        return self.actions[label_index]

    def get_label_from_action(self, action_name: str) -> int:
        return [label_index for label_index, val in self.actions.items() if v == action_name][0]


if __name__ == "__main__":
    # set up
    dataset = Dataset()

    # execute
    dataset.load()
