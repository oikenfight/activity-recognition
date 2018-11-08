from datetime import datetime
import os
import sys
import glob
import time
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from converter import Converter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class ConverterBatch:
    """
    時系列画像からフレームを生成、フレームの画像を読み込み、画像リサイズ、pkl 保存。
    """

    # constant
    BASE = './src/3dcnn_lstm_model/img2vec'
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self, input_base_dir, output_base_dir):
        self.input_base_dir = input_base_dir                # データの場所は docker-compose でコンテナにマウントすること。
        self.output_base_dir = output_base_dir    # output の場所もコンテナから見える場所を指定すること。
        self.target_key = ''
        self.file_manager = None

    def _prepare(self):
        # ファイルを取得する準備
        FileManager.BASE_DIR = self.input_base_dir
        self.file_manager = FileManager()

        # 保存先ディレクトリを作成
        self.target_key = datetime.now().strftime("%Y%m%d_%H%M%S")

    def main(self):
        print('<<< ConverterBatch class: main method >>>')
        self._prepare()

        all_dir_list = self.file_manager.all_dir_lists()
        total = len(all_dir_list)

        # setup Converter
        converter_instance = Converter()

        for num, dir_path_in_list in enumerate(all_dir_list):
            """
            dir_path_in_list[0]: datetime
            dir_path_in_list[1]: アクション名
            dir_path_in_list[2]: ビデオ名
            """
            input_dir = self.input_base_dir + '/'.join(dir_path_in_list) + '/'
            output_dir = self.output_base_dir + self.target_key + "/" + dir_path_in_list[1] + '/'
            output = output_dir + dir_path_in_list[2]
            image_files = sorted(glob.glob(input_dir + '/*.jpg'))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # フレームを作成して、ラベルと共に追加
            for i in range(0, len(image_files), self.OVERLAP_SIZE):
                frame = image_files[i:i + self.FRAME_SIZE]
                # TODO: 微妙に 8 フレームに足りてないデータを無駄にしてるから、できたら直す（学習する時にフレーム数が一定のほうが楽だから今はこうしてる。）
                if len(frame) < 8:
                    break

                images_vec = []
                for image_path in frame:
                    image_vec = converter_instance.main(image_path)
                    images_vec.append(image_vec)

                output_file_path = '%s_%s_%s.pkl' % (output, str(i), str(i + self.FRAME_SIZE))
                self._dump_to(output_file_path, images_vec)
                print("%s / %s output: %s" % (str(num), str(total), output_file_path))

    @staticmethod
    def _dump_to(path, data):
        print('dumping data shape: (', str(len(data)), str(len(data[0])), str(len(data[0][0])), str(len(data[0][0][0])), ')')
        with open(path, 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #
    # Example
    #

    # params
    input_base_dir = '/converted_data/'
    output_base_dir = '/image_vec_data/'

    # execute
    converter_batch = ConverterBatch(input_base_dir, output_base_dir)
    converter_batch.main()
