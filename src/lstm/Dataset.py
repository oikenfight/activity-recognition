import pickle as pkl
import numpy as np
from chainer.datasets import tuple_dataset
from FileManager import FileManager
import time


class Dataset:
    # TASK: 要セットアップ
    BASE = './src/lstm/'
    INPUT_FILES_BASE = './src/cnn/dataset/20180815_065125'
    FileManager.BASE_DIR = INPUT_FILES_BASE
    FEATURE_SIZE = 4096
    file_manager = FileManager()

    def __init__(self):
        self.all_files = self.file_manager.all_files()  # List(str)
        self.actions = {}
        self.label_data = np.empty((0, self.get_output_size()), int)
        self.features_data = np.empty((0, 11, self.FEATURE_SIZE), float)  # TODO: 11 直書きキモいから直す
        self.load()

    def load(self):
        print('Dataset Loading ...')
        print('action length: ', str(len(self.all_files)))
        for i, pkl_file in enumerate(self.all_files):
            print('=================================')
            print(pkl_file)

            # actions 辞書作成
            action_name = pkl_file.split('.')[0]
            self.actions[len(self.actions)] = action_name

            # 正解ラベル作成
            action_label = np.zeros(len(self.all_files))
            action_label[i] = 1

            print(action_label)

            with open(pkl_file, 'rb') as f:
                dataset = pkl.load(f)
            print('data number: ', str(len(dataset)))

            # ラベルと時系列画像特徴量リストが対応するデータセットを作成
            for k, (folder_name, features) in enumerate(dataset.items()):
                self.label_data = np.append(self.label_data, np.array([action_label]), axis=0)
                # TODO: 切り出し間隔ちゃんと考えたほうがいいかな。。 _working_memo に書いた通りに。
                self.features_data = np.append(self.features_data, np.array([features[:11]]), axis=0)

    def get_output_size(self) -> int:
        return len(self.all_files)

    def get_action_from_label(self, label_index: int) -> str:
        return self.actions[label_index]

    def get_label_from_action(self, action_name: str) -> int:
        return [label_index for label_index, val in self.actions.items() if val == action_name][0]

    def shuffle_to_train_test(self, test_rate: float) -> tuple:
        """
        任意の割合でランダムに訓練データとテストデータの tuple を返却する
        :param float test_rate: テストデータにする割合
        :return tuple (train, test):
            (
                ([features, features, ...], [label, label, ...]),
                ([features, features, ...], [label, label, ...])
            )
        """
        print()
        print('<<<< shuffle data to train and test >>>>')

        test_num = int(len(self.label_data) * test_rate)
        indexes = np.array(range(len(self.label_data)))
        np.random.shuffle(indexes)
        train_indexes = indexes[test_num:]
        test_indexes = indexes[:test_num]

        # TODO: 11 直書きはキモいから直す
        train_data, train_label = np.empty((0, 11, self.FEATURE_SIZE), float), np.empty((0, self.get_output_size()), int)
        test_data, test_label = np.empty((0, 11, self.FEATURE_SIZE), float), np.empty((0, self.get_output_size()), int)

        for i in train_indexes:
            train_data = np.append(train_data, np.array([self.features_data[i]]), axis=0)
            train_label = np.append(train_label, np.array([self.label_data[i]]), axis=0)
        for i in test_indexes:
            test_data = np.append(test_data, np.array([self.features_data[i]]), axis=0)
            test_label = np.append(test_label, np.array([self.label_data[i]]), axis=0)

        return (train_data.astype(np.float32), train_label.astype(np.int32)),\
               (test_data.astype(np.float32), test_label.astype(np.int32))


if __name__ == "__main__":
    # set up
    dataset = Dataset()
