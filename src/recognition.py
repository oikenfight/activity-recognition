from datetime import datetime
import os
import pickle as pkl
from FileManager import FileManager
from cnn_activity_recognizer import CnnActivityRecognizer
from lrcn_activity_recognizer import LrcnActivityRecognizer
import threading
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
    CNN_MODEL_PATH  = ''
    LRCN_MODEL_PATH = ''
    PERSON_LRCN_MODEL_PATH = ''
    THREAD_SIZE = 5

    # constant
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3
    FRAME_SIZE = 8
    OVERLAP_SIZE = 4

    def __init__(self):
        self.target_key = ''
        self.file_manager = None
        self.actions = None
        self.recognition_results = None
        self.cnn_activity_recognizer_instance = None
        self.lrcn_activity_recognizer_instance = None
        self.person_lrcn_activity_recognizer_instance = None
        self.recognition_results = {}
        self.current_label = None
        self.current_action = None

    def _prepare(self):
        self._set_file_manager()
        self._load_actions()
        self._set_recognizer()
        self._init_recognition_results()

    def _set_file_manager(self):
        # ファイルを取得する準備
        # FileManager.BASE_DIR = self.INPUT_PERSON_IMAGES_DIR
        FileManager.BASE_DIR = self.INPUT_IMAGES_DIR
        self.file_manager = FileManager()
        # 保存先ディレクトリを作成
        self.target_key = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_actions(self):
        with open(self.ACTIONS_PATH, mode="rb") as f:
            self.actions = pkl.load(f)
        print('action length: ', str(len(self.actions)))
        print(self.actions)

    def _set_recognizer(self):
        # cnn activity recognizer instance
        CnnActivityRecognizer.MODEL_PATH = self.CNN_MODEL_PATH
        # self.cnn_activity_recognizer_instance = CnnActivityRecognizer()

        # lrcn activity recognizer instance
        LrcnActivityRecognizer.MODEL_PATH = self.LRCN_MODEL_PATH
        self.lrcn_activity_recognizer_instance = LrcnActivityRecognizer(self.actions)

        # person lrcn activity recognizer instance
        LrcnActivityRecognizer.MODEL_PATH = self.PERSON_LRCN_MODEL_PATH
        # self.person_lrcn_activity_recognizer_instance = LrcnActivityRecognizer()

    def _init_recognition_results(self):
        """
        認識結果データを格納。各データにつき構造は以下の通り。
        {
            '{action_name}': {
                '{video_name}': [
                    {
                        'cnn': ({max_index}, {max_value}, {actual_index}, {actual_value}),
                        'lrcn': ({max_index}, {max_value}, {actual_index}, {actual_value}),
                        'person_lrcn': ({max_index}, {max_value}, {actual_index}, {actual_value}),
                    },
                    ...
                ]
            },
            ...
        }
        :return:
        """
        for label, action_name in self.actions.items():
            self.recognition_results[action_name] = {}

    def _append_video_recognition_result(self, action_name: str, video_name: str, results: dict):
        # TODO: add cnn and person_lrcn result
        print('_lrcn_result', results['lrcn'])
        if not self.recognition_results[action_name].get(video_name):
            self.recognition_results[action_name][video_name] = [results]
        else:
            self.recognition_results[action_name][video_name].append(results)

    def main(self):
        print('<<< Recognition class: main method >>>')
        self._prepare()

        all_dir_with_list = self.file_manager.all_dir_lists()
        total = len(all_dir_with_list)

        threads_data = []
        for num, dir_path_with_list in enumerate(all_dir_with_list):
            """
            dir_path_with_list[0]: アクション名
            dir_path_with_list[1]: ビデオ名
            """
            self.current_label = self._get_action_label(dir_path_with_list[0])
            # input_person_dir = self.INPUT_PERSON_IMAGES_DIR + '/'.join(dir_path_with_list) + '/'
            input_image_dir = self.INPUT_IMAGES_DIR + '/'.join(dir_path_with_list) + '/'
            # images = sorted(os.listdir(input_person_dir))
            images = sorted(os.listdir(input_image_dir))

            # フレームを作成して、ラベルと共に追加
            for i in range(0, len(images), self.OVERLAP_SIZE):
                frame_images = images[i:i+self.FRAME_SIZE]
                # TODO: 微妙に 8 フレームに足りてないデータを無駄にしてるから、できたら直す（学習する時にフレーム数が一定のほうが楽だから今はこうしてる。）
                if len(frame_images) < 8:
                    break

                # person_frame = [input_person_dir + filename for filename in frame_images]
                image_frame = [input_image_dir + filename for filename in frame_images]

                # スレッドで推測を実行
                if len(threads_data) < self.THREAD_SIZE:
                    # threads_data.append(frame_images, person_frame)
                    threads_data.append((image_frame, dir_path_with_list[0], dir_path_with_list[1]))
                else:
                    self._do_recognition(threads_data)
                    print(num, '/', total, 'is completed.')
                    threads_data = []
        if not threads_data == []:
            self._do_recognition(threads_data)
            threads_data = []
        self._dump_to()

    def _get_action_label(self, action_name: str) -> int:
        # TODO: actions.pkl から読み込んだ辞書をつかって、アクション名から正解ラベルを取得
        keys = [key for key, value in self.actions.items() if action_name == value]
        if len(keys) == 1:
            return keys[0]
        else:
            print('おかしい。action_name が actions 辞書に登録されてないぞ。。')

    def _do_recognition(self, thread_data: list):
        """
        スレッドを実行。
        :param thread_data: list of tuple of (image_frame, action_name, video_name)
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
        return

    def _recognize_thread(self, data: tuple):
        # 各手法で認識を行う
        action_name = data[1]
        video_name = data[2]

        # _cnn_acc = self._cnn_recognize(data[0])
        _lrcn_result = self._lrcn_recognize(data[0])
        # _person_lrcn_result = self._person_lrcn_recognize(data[0])

        results = {
            'cnn': (),
            'lrcn': _lrcn_result,
            'person_lrcn': (),
        }
        self._append_video_recognition_result(action_name, video_name, results)

    def _cnn_recognize(self, image_frame: list):
        max_index, max_value, actual_label, actual_value = self.cnn_activity_recognizer_instance.main(image_frame)
        return max_index, max_value, actual_label, actual_value

    def _lrcn_recognize(self, image_frame: list):
        max_index, max_value, actual_label, actual_value = self.lrcn_activity_recognizer_instance.main(image_frame, self.current_label)
        return max_index, max_value, actual_label, actual_value

    def _person_lrcn_recognize(self, person_frame: list, label: int):
        max_index, max_value, actual_label, actual_value = self.person_lrcn_activity_recognizer_instance.main(person_frame)
        return max_index, max_value, actual_label, actual_value

    def _dump_to(self):
        save_path = './' + self.target_key + '_result.pkl'
        print('dump to result as : ' + save_path)

        # 保存されているデータを一旦削除
        if os.path.isfile(save_path):
            os.remove(save_path)

        with open(save_path, 'wb') as f:
            pkl.dump(self.recognition_results, f, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # set up
    Recognition.INPUT_PERSON_IMAGES_DIR = '/person_images_data/{datetime}/'
    # Recognition.INPUT_IMAGES_DIR = '/images_data/20181204_013516/'
    Recognition.INPUT_IMAGES_DIR = '/images_data/test/'
    Recognition.ACTIONS_PATH = 'output/lrcn_recognition/models/20181206_202352/actions.pkl'
    Recognition.CNN_MODEL_PATH  = ''
    Recognition.LRCN_MODEL_PATH = 'output/lrcn_recognition/models/20181206_202352/0140.model'
    Recognition.PERSON_LRCN_MODEL_PATH = ''
    # params
    # instance
    recognition_instance = Recognition()
    recognition_instance.main()
