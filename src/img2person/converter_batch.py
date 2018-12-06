from datetime import datetime
import os
import sys
import glob
import time
import pickle as pkl
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from converter import Converter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class ConverterBatch:
    """
    時系列画像データから人物検出を行い、その部分を切り出しつつ、
    アスペクト比を保ちながら正方形にサイズ変換して保存する
    """

    # constant
    BASE = './src/img2person'

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
            output_action_dir = self.output_base_dir + self.target_key + "/" + dir_path_in_list[1] + '/'
            output_video_dir = output_action_dir + dir_path_in_list[2] + '/'
            image_files = sorted(glob.glob(input_dir + '/*.jpg'))

            if not os.path.exists(output_action_dir):
                os.makedirs(output_action_dir)
            os.mkdir(output_video_dir)

            incompleted_num = 0
            # フレームを作成して、ラベルと共に追加
            for input_path in image_files:
                img_name = input_path.split('/')[-1]
                output_path = output_video_dir + img_name

                try:
                    converter_instance.main(input_path, output_path)
                except:
                    print('人物捻出を検出できなかったので飛ばします。')
                    incompleted_num += 1
                if incompleted_num > 3:
                    shutil.rmtree(output_video_dir)
                    print('人物検出が明確ではないため削除しました。')
                    break
            print("%s / %s output: %s" % (str(num), str(total), output_video_dir))

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
    output_base_dir = '/person_data/'

    # execute
    converter_batch = ConverterBatch(input_base_dir, output_base_dir)
    converter_batch.main()
