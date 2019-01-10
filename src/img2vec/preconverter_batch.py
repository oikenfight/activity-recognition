from datetime import datetime
import os
import sys
import glob
import time
import pickle as pkl

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from preconverter import PreConverter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class PreConverterBatch:
    """
    時系列画像データをアスペクト比を保ちながら正方形にサイズ変換して保存する
    """

    # constant
    BASE = './src/img3vec'

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
        print('<<< PreConverterBatch class: main method >>>')
        self._prepare()

        all_dir_list = self.file_manager.all_dir_lists()
        total = len(all_dir_list)
        print(total)

        # setup Converter
        preconverter_instance = PreConverter()

        for num, dir_path_in_list in enumerate(all_dir_list):
            """
            dir_path_in_list[0]: datetime
            dir_path_in_list[1]: アクション名
            dir_path_in_list[2]: ビデオ名
            """
            input_dir = self.input_base_dir + '/'.join(dir_path_in_list) + '/'
            output_action_dir = self.output_base_dir + self.target_key + "/" + dir_path_in_list[1] + '/'
            output_video_dir = output_action_dir + dir_path_in_list[2] + '/'
            image_files = sorted(glob.glob(input_dir + '/*.jpg'))

            if not os.path.exists(output_action_dir):
                os.makedirs(output_action_dir)
            os.mkdir(output_video_dir)

            # フレームを作成して、ラベルと共に追加
            for input_path in image_files:
                img_name = input_path.split('/')[-1]
                output_path = output_video_dir + img_name

                preconverter_instance.main(input_path, output_path)

            print("%s / %s output: %s" % (str(num), str(total), output_video_dir))


if __name__ == "__main__":
    #
    # Example
    #

    # params
    input_base_dir = '/images_data/'
    output_base_dir = '/resized_images_data/'

    # execute
    converter_batch = PreConverterBatch(input_base_dir, output_base_dir)
    converter_batch.main()

