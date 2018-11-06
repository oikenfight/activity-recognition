from datetime import datetime
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from converter import Converter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class ConverterBatch:
    # constant
    BASE = './src/frames'

    # setup
    FPS = 2

    def __init__(self, input_dir, output_base_dir):
        self.input_dir = input_dir                # データの場所は docker-compose でコンテナにマウントすること。
        self.output_base_dir = output_base_dir    # output の場所もコンテナから見える場所を指定すること。
        self.target_key = ''
        self.file_manager = None
        self.all_file_lists = []

    def _prepare(self):
        # ファイルを取得する準備
        FileManager.BASE_DIR = self.input_dir
        self.file_manager = FileManager()

        # 保存先ディレクトリを作成
        self.target_key = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_base_dir + self.target_key + '/')

    def main(self):
        print('<<< ConverterBatch class: main method >>>')
        self._prepare()

        all_file_list = self.file_manager.all_file_lists()
        total = len(all_file_list)

        # setup Converter
        Converter.FPS = self.FPS
        converter_instance = Converter()

        for i, file_list in enumerate(all_file_list):
            video_file_name_without_extension = file_list[1].split('.')[0]  # video_file_name の拡張子を覗いたものをディレクトリ名にする

            # params for converter
            input_path = self.input_dir + file_list[0] + '/' + file_list[1]
            output_dir = self.output_base_dir + self.target_key + '/' + file_list[0] + '/' + video_file_name_without_extension + '/'

            # flame_data/{action}/{file_name} ディレクトリを作成
            os.makedirs(output_dir)

            # execute
            converter_instance.main(input_path, output_dir)

            print("%s / %s output: %s" % (str(i), str(total), output_dir))


if __name__ == "__main__":
    #
    # Example
    #

    # setup
    ConverterBatch.FPS = 2

    # params
    input_dir = '/data/STAIR-actions/stair_action/'
    output_dir = '/converted_data/'

    # execute
    converter_batch = ConverterBatch(input_dir, output_dir)
    converter_batch.main()
