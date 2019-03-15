from datetime import datetime
import os
import sys
import glob
import threading
import time
# import pickle as pkl
import shutil
from pprint import pprint as pp

from multiprocessing import Pool
import multiprocessing as multi

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from converter import Converter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


completed_num = 0
data_length = 0


class ConverterBatch:
    """
    時系列画像データから人物検出を行い、その部分を切り出しつつ、
    アスペクト比を保ちながら正方形にサイズ変換して保存する
    """

    # constant
    BASE = './src/img2person'
    THREAD_SIZE = 20

    def __init__(self, input_base_dir, output_base_dir):
        self.input_base_dir = input_base_dir                # データの場所は docker-compose でコンテナにマウントすること。
        self.output_base_dir = output_base_dir    # output の場所もコンテナから見える場所を指定すること。
        self.converter_instance = None
        self.target_key = ''
        self.file_manager = None
        self.log_file = None

    def _prepare(self):
        # ファイルを取得する準備
        FileManager.BASE_DIR = self.input_base_dir
        self.file_manager = FileManager()

        # 保存先ディレクトリを作成
        self.target_key = datetime.now().strftime("%Y%m%d_%H%M%S")

        # # setup Converter
        # self.converter_instance = Converter()

    def main(self):
        print('<<< ConverterBatch class: main method >>>')
        self._prepare()

        global data_length

        all_dir_list = self.file_manager.all_dir_lists()
        total = len(all_dir_list)

        data_length = len(all_dir_list)
        all_threads_data = []
        threads_data = []

        for num, dir_path_in_list in enumerate(all_dir_list):
            """
            dir_path_in_list[0]: アクション名
            dir_path_in_list[1]: ビデオ名
            """

            # if dir_path_in_list[0] in stop_action:
            #     continue

            input_dir = self.input_base_dir + '/'.join(dir_path_in_list) + '/'
            output_action_dir = self.output_base_dir + self.target_key + "/" + dir_path_in_list[0] + '/'
            output_video_dir = output_action_dir + dir_path_in_list[1] + '/'
            image_files = sorted(glob.glob(input_dir + '/*.jpg'))

            if not os.path.exists(output_action_dir):
                os.makedirs(output_action_dir)
            os.mkdir(output_video_dir)

            threads_data.append((image_files, output_video_dir))
            if len(threads_data) >= self.THREAD_SIZE:
                start = datetime.now()
                # self._do_detect(threads_data)
                # print(num, '/', data_length, 'is completed.', 'time:', (datetime.now() - start))
                all_threads_data.append(threads_data)
                threads_data = []
                # self._open_log_file()
                # self._write_line_to_log(str(num) + '/' + str(data_length) + 'is completed. time:' + str((datetime.now() - start)))
                # self._close_log_file()
        else:
            all_threads_data.append(threads_data)
            # self._do_detect(threads_data)

        p = Pool(8)
        p.map(_do_detect, all_threads_data)
        p.close()

    # def _do_detect(self, thread_data: list):
    #     """
    #     スレッドを実行。
    #     :param thread_data: list of tuple of (image_files, output_video_dir)
    #     :return:
    #     """
    #     print('do_detect')
    #     thread_list = []
    #     # スレッドをセット
    #     for data in thread_data:
    #         thread = threading.Thread(target=self._detect_thread,
    #                                   args=([data]))
    #         thread_list.append(thread)
    #     # スレッド実行
    #     for thread in thread_list:
    #         thread.start()
    #     # 全てのスレッドが完了するのを待機
    #     for thread in thread_list:
    #         thread.join()
    #     return
    #
    # def _detect_thread(self, data):
    #     image_files = data[0]
    #     output_video_dir = data[1]
    #     incompleted_num = 0
    #     # フレームを作成して、ラベルと共に追加
    #     for input_path in image_files:
    #         # print(input_path)
    #         path_data = input_path.split('/')
    #         img_name = path_data[-1]
    #         output_path = output_video_dir + img_name
    #
    #         try:
    #             self.converter_instance.main(input_path, output_path)
    #         except:
    #             print('人物を検出できなかったので飛ばします。')
    #             incompleted_num += 1
    #         if incompleted_num > 3:
    #             if os.path.isdir(output_video_dir):
    #                 shutil.rmtree(output_video_dir)
    #                 print('人物検出が明確ではないため削除しました。')
    #             break

    # def _open_log_file(self):
    #     path = './tmp/img2person_batch.log'
    #     self.log_file = open(path, mode='a')
    #
    # def _write_line_to_log(self, s: str):
    #     self.log_file.write(s + '\n')
    #
    # def _close_log_file(self):
    #     self.log_file.close()


# クラスメソッドだと並列化できないから仕方なく。
def _do_detect(thread_data: list):
    """
    スレッドを実行。
    :param thread_data: list of tuple of (image_files, output_video_dir)
    :return:
    """
    # setup Converter
    converter_instance = Converter()

    thread_list = []
    # スレッドをセット
    for data in thread_data:
        thread = threading.Thread(target=_detect_thread,
                                  args=([data, converter_instance]))
        thread_list.append(thread)

    # スレッド実行
    for thread in thread_list:
        thread.start()
    # 全てのスレッドが完了するのを待機
    for thread in thread_list:
        thread.join()

    global completed_num
    completed_num += ConverterBatch.THREAD_SIZE
    update_log(str(completed_num) + '/' + str(data_length) + ' is completed')
    return


def _detect_thread(data, converter_instance):
    image_files = data[0]
    output_video_dir = data[1]
    incompleted_num = 0
    # フレームを作成して、ラベルと共に追加
    for input_path in image_files:
        # print(input_path)
        path_data = input_path.split('/')
        img_name = path_data[-1]
        output_path = output_video_dir + img_name

        try:
            time.sleep(3)
            converter_instance.main(input_path, output_path)
        except:
            # print('人物を検出できなかったので飛ばします。')
            incompleted_num += 1
        if incompleted_num > 5:
            if os.path.isdir(output_video_dir):
                shutil.rmtree(output_video_dir)
                update_log('人物検出が明確ではないため削除しました。' + output_video_dir)
            break
    else:
        update_log('save' + output_video_dir)


def update_log(s: str):
    path = './tmp/re_img2person_batch.log'
    log_file = open(path, mode='a')
    log_file.write(s + '\n')
    log_file.close()


if __name__ == "__main__":
    #
    # Example
    #

    # params
    input_base_dir = '/images_data/20181229_120953/'
    output_base_dir = '/person_images_data/'

    # execute
    converter_batch = ConverterBatch(input_base_dir, output_base_dir)
    converter_batch.main()
