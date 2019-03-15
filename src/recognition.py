from datetime import datetime
import pickle as pkl
from cnn_activity_recognizer import CnnActivityRecognizer
from lrcn_activity_recognizer import LrcnActivityRecognizer
import threading
import numpy as np
import time
from pprint import pprint as pp


class Recognition:
    """
    全てのデータを用いて行動認識を行い、それぞれの手法の認識精度を確かめる。
    ただし、対象データは人物検出が可能だったもののみとする。（精度の比較を行うため、完全に同一のデータのみを用いたい。）
    まず、フレーム毎に行動の推測を行い、各フレームごとに認識精度を確かめる。
    次に、各アクションごとに認識精度を平均し、比較できるようにする。
    """

    # set up
    INPUT_PERSON_IMAGES_DIR = '/person_images_data/{datetime}/'
    INPUT_IMAGES_DIR = '/images_data/{datetime}/'
    ACTIONS_PATH = ''
    TEST_DATA_FILE = ''
    MODEL_PATH = ''
    METHOD_TYPE = ''  # cnn or lrcn or person_lrcn
    THREAD_SIZE = 8

    # constant
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self):
        self.actions = {}
        self.test_data_paths = []
        self.recognizer_instance = None
        self.recognized_result = {}
        self.log_file = None

    def _prepare(self):
        self._load_actions()
        self._load_test_data()
        self._set_recognizer()

    def _load_actions(self):
        with open(self.ACTIONS_PATH, mode="rb") as f:
            self.actions = pkl.load(f)
        print('action length: ', str(len(self.actions)))
        print(self.actions)

    def _load_test_data(self):
        with open(self.TEST_DATA_FILE, mode="rb") as f:
            self.test_data_paths = pkl.load(f)
        print('test data length: ', str(len(self.test_data_paths)))

    def _set_recognizer(self):
        # TODO: ここは手動で切り替えることにしよう。ちょっと時間がない

        # # cnn activity recognizer instance
        # CnnActivityRecognizer.MODEL_PATH = self.MODEL_PATH
        # self.recognizer_instance = CnnActivityRecognizer(self.actions)

        # lrcn activity recognizer instance
        LrcnActivityRecognizer.MODEL_PATH = self.MODEL_PATH
        self.recognizer_instance = LrcnActivityRecognizer(self.actions)

        # person lrcn activity recognizer instance
        # LrcnActivityRecognizer.MODEL_PATH = self.MODEL_PATH
        # self.recognizer_instance = LrcnActivityRecognizer(self.actions)

    def main(self):
        print('<<< Recognition class: main method >>>')
        self._prepare()

        data_length = len(self.test_data_paths)
        print(data_length)
        threads_data = []
        for i, frame in enumerate(self.test_data_paths):
            action_name = frame[0].split('/')[3]
            video_name = frame[0].split('/')[4]
            action_label = self._get_action_label(action_name)

            threads_data.append((frame, action_name, video_name, action_label))
            if len(threads_data) >= self.THREAD_SIZE:
                start = datetime.now()
                self._do_recognition(threads_data)
                print(i, '/', data_length, 'is completed.', 'time:', (datetime.now() - start))
                self._open_log_file()
                self._write_line_to_log(str(i) + '/' + str(data_length) + 'is completed. time:' + str((datetime.now() - start)))
                self._close_log_file()
                threads_data = []

            # if i > 1001:
            #     break
        else:
            self._do_recognition(threads_data)

        self._dump_to()

    def _get_action_label(self, action_name: str) -> int:
        # actions.pkl から読み込んだ辞書をつかって、アクション名から正解ラベルを取得
        keys = [key for key, value in self.actions.items() if action_name == value]
        if len(keys) == 1:
            return keys[0]
        else:
            print('おかしい。action_name が actions 辞書に登録されてないぞ。。')

    def _do_recognition(self, thread_data: list):
        """
        スレッドを実行。
        :param thread_data: list of tuple of (frame, action_name, video_name, action_label)
        :return:
        """
        thread_list = []
        # スレッドをセット
        for data in thread_data:
            thread = threading.Thread(target=self._recognize_thread,
                                      args=([data]))
            thread_list.append(thread)
        # スレッド実行
        for thread in thread_list:
            thread.start()
        # 全てのスレッドが完了するのを待機
        for thread in thread_list:
            thread.join()
        # print('did recognition')
        return

    def _recognize_thread(self, data: tuple):
        # 各手法で認識を行う
        frame = data[0]
        action_name = data[1]
        video_name = data[2]
        action_label = data[3]
        try:
            result = self.recognizer_instance.main(frame, action_label)
        except Exception as e:
            print(e)
            self._open_log_file()
            self._write_line_to_log(action_name + '/' + video_name + 'でなんかエラー吐いた')
            self._close_log_file()

        # print('max_label:', str(result[0]), 'max_value:', str(result[1]), 'actual_label:', str(result[2]), 'actual_value:', result[3])

        if not self.recognized_result.get(action_name):
            # 新規アクションの場合
            self.recognized_result[action_name] = {}
        if not self.recognized_result[action_name].get(video_name):
            # 新規ビデオの場合
            self.recognized_result[action_name][video_name] = [result]
        else:
            # 同じビデオの別フレームが存在する場合
            self.recognized_result[action_name][video_name].append(result)

    def _dump_to(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = './' + current_time + '_' + self.METHOD_TYPE + '_result.pkl'
        print('dump to result as : ' + save_path)

        with open(save_path, 'wb') as f:
            pkl.dump(self.recognized_result, f, pkl.HIGHEST_PROTOCOL)

    def _open_log_file(self):
        path = './tmp/recognition.log'
        self.log_file = open(path, mode='a')

    def _write_line_to_log(self, s: str):
        self.log_file.write(s + '\n')

    def _close_log_file(self):
        self.log_file.close()


if __name__ == '__main__':
    #
    # set up
    #
    Recognition.ACTIONS_PATH = 'output/lrcn_recognition/models/20190101_063332/actions.pkl'
    Recognition.TEST_DATA_FILE = 'output/lrcn_recognition/models/20190101_063332/test_frame_data.pkl'
    Recognition.MODEL_PATH = 'output/lrcn_recognition/models/20190101_063332/0159.model'
    Recognition.METHOD_TYPE = 'lrcn'

    # params
    # instance
    recognition_instance = Recognition()
    recognition_instance.main()
