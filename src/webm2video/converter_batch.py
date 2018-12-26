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
    BASE = './src/3dcnn_lstm_model/webm2video'

    # setup
    FRAME_RATE = 30

    def __init__(self, input_base_dir, output_base_dir):
        self.input_base_dir = input_base_dir                # データの場所は docker-compose でコンテナにマウントすること。
        self.output_base_dir = output_base_dir    # output の場所もコンテナから見える場所を指定すること。
        self.target_key = ''
        self.file_manager = None
        self.all_file_lists = []

    def _prepare(self):
        # ファイルを取得する準備
        FileManager.BASE_DIR = self.input_base_dir
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
        Converter.FRAME_RATE = self.FRAME_RATE
        converter_instance = Converter()

        for i, file_list in enumerate(all_file_list):
            # params for converter
            input_path = self.input_base_dir + file_list[0] + '/' + file_list[1]
            output_dir = self.output_base_dir + self.target_key + '/' + file_list[0] + '/'

            # execute
            converter_instance.main(input_path, output_dir)

            print("%s / %s output: %s" % (str(i), str(total), output_dir))


if __name__ == "__main__":
    #
    # Example
    #

    # setup
    ConverterBatch.FRAME_RATE = 30

    # params
    input_dir = '/sentences/original-actions/webm/actions/'
    output_dir = '/sentences/original-actions/actions/'

    # execute
    converter_batch_instance = ConverterBatch(input_dir, output_dir)
    converter_batch_instance.main()
