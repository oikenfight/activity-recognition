import pickle as pkl
from chainer import cuda
import time
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class FrameCreator:
    """
    指定したディレクトリ内の pkl ファイルを全て読み込み、
    クラス変数で指定した FRAME_SIZE に分割した 1 つの pkl ファイルを生成する。
    label と features のデータは対応。
    ex)
    output_pkl_data = {
        'label': [0, 0, 0, ..., 1, 1, 1, ..., n, n, n],
        'features': [
                [[cnn_vec_0], [cnn_vec_1], ..., [cnn_vec_FEATURE_SIZE]],
                [[cnn_vec_0], [cnn_vec_1], ..., [cnn_vec_FEATURE_SIZE]],
                ...
            ],
        'actions_dict': {
                '0': 'some action 0',
                '1': 'some action 1',
                ...
                'n': 'some action N',
            },
    }
    """

    # constant
    BASE = './src/lstm/'

    # setup
    FEATURE_SIZE = 4096
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self, input_cnn_dir: str, output_dir: str):
        self.input_cnn_dir = input_cnn_dir
        self.output_dir = output_dir
        self.actions = {}
        self.features_data = []
        self.label_data = []

    @staticmethod
    def _load_all_pkl_files(input_cnn_dir):
        FileManager.BASE_DIR = input_cnn_dir
        file_manager = FileManager()
        return file_manager.all_files()

    def main(self):
        print('Dataset Loading ...')
        all_pkl_files = self._load_all_pkl_files(self.input_cnn_dir)
        print('action length: ', str(len(all_pkl_files)))

        for i, pkl_file in enumerate(all_pkl_files):
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
                    self.features_data += [frame_data]

                if k % 50 == 0:
                    print(str(k), '/', len(dataset), ':', folder_name, ', data:(', len(self.features_data), len(self.features_data[0]), len(self.features_data[0][0]), '), label:', len(self.label_data))
        # dump to
        self._dump_to_framed_pkl()

    def _output_file_name(self):
        """
        入力したディレクトリの最後（日付部分）を引用して出力ファイル名を決定する
        :return str:
        """
        # expected self.input_cnn_dir: '/path/to/output/cnn/{datetime}/'
        base_cnn_dir_name = self.input_cnn_dir.split('/')[-2]
        return self.output_dir + base_cnn_dir_name + '.pkl'

    def _dump_to_framed_pkl(self):
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
    #
    # Example
    #

    # setup
    # params
    input_cnn_dir = './output/cnn/20180929_075743/'
    output_dir = './output/framed_cnn/'

    # execute
    frame_creator_instance = FrameCreator(input_cnn_dir, output_dir)
    frame_creator_instance.main()



