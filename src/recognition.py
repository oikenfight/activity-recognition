from datetime import datetime
import os
import pickle as pkl
from FileManager import FileManager
from cnn_activity_recognizer import CnnActivityRecognizer
from lrcn_activity_recognizer import LrcnActivityRecognizer


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
        self.cnn_activity_recognizer_instance = None
        self.lrcn_activity_recognizer_instance = None
        self.person_lrcn_activity_recognizer_instance = None
        self.recognition_results = {}

    def _prepare(self):
        self._set_file_manager()
        self._load_actions()
        self._set_recognizer()
        self._init_recognition_results()

    def _set_file_manager(self):
        # ファイルを取得する準備
        FileManager.BASE_DIR = self.INPUT_PERSON_IMAGES_DIR
        self.file_manager = FileManager()
        # 保存先ディレクトリを作成
        self.target_key = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_actions(self):
        with open(self.ACTIONS_PATH, mode="rb") as f:
            self.actions = pkl.load(f)

    def _set_recognizer(self):
        # cnn activity recognizer instance
        CnnActivityRecognizer.MODEL_PATH = self.CNN_MODEL_PATH
        self.cnn_activity_recognizer_instance = CnnActivityRecognizer()

        # lrcn activity recognizer instance
        LrcnActivityRecognizer.MODEL_PATH = self.LRCN_MODEL_PATH
        self.lrcn_activity_recognizer_instance = LrcnActivityRecognizer()

        # person lrcn activity recognizer instance
        LrcnActivityRecognizer.MODEL_PATH = self.PERSON_LRCN_MODEL_PATH
        self.person_lrcn_activity_recognizer_instance = LrcnActivityRecognizer()

    def _init_recognition_results(self):
        # TODO: 各アクションごとに各フレームの正答率をリストで格納する
        pass

    def main(self):
        print('<<< Recognition class: main method >>>')
        self._prepare()

        all_dir_with_list = self.file_manager.all_dir_lists()
        total = len(all_dir_with_list)

        for num, dir_path_with_list in enumerate(all_dir_with_list):
            """
            dir_path_with_list[0]: アクション名
            dir_path_with_list[1]: ビデオ名
            """
            label = self._get_action_label(dir_path_with_list[0])
            input_person_dir = self.INPUT_PERSON_IMAGES_DIR + '/'.join(dir_path_with_list) + '/'
            input_image_dir = self.INPUT_IMAGES_DIR + '/'.join(dir_path_with_list) + '/'
            images = sorted(os.listdir(input_person_dir))

            # フレームを作成して、ラベルと共に追加
            for i in range(0, len(images), self.OVERLAP_SIZE):
                frame_images = images[i:i+self.FRAME_SIZE]
                # TODO: 微妙に 8 フレームに足りてないデータを無駄にしてるから、できたら直す（学習する時にフレーム数が一定のほうが楽だから今はこうしてる。）
                if len(frame_images) < 8:
                    break
                person_frame = [input_person_dir + filename for filename in frame_images]
                image_frame = [input_image_dir + filename for filename in frame_images]
                # 各手法で認識を行う
                # TODO: 認識して、結果はどうやって集計しようか。。
                _cnn_acc = self._cnn_recognize(image_frame, label)
                _lrcn_acc = self._lrcn_recognize(image_frame, label)
                _person_lrcn_acc = self._person_lrcn_recognize(person_frame, label)

    def _get_action_label(self, action_name: str) -> int:
        # TODO: actions.pkl から読み込んだ辞書をつかって、アクション名から正解ラベルを取得
        return self.actions[action_name]

    def _cnn_recognize(self, image_frame: list, label: int):
        acc, loss = self.cnn_activity_recognizer_instance.main(image_frame, label)
        return acc

    def _lrcn_recognize(self, image_frame: list, label: int):
        acc, loss = self.lrcn_activity_recognizer_instance.main(image_frame, label)
        return acc

    def _person_lrcn_recognize(self, person_frame: list, label: int):
        acc, loss = self.person_lrcn_activity_recognizer_instance.main(person_frame, label)
        return acc


if __name__ == '__main__':
    # set up
    Recognition.INPUT_PERSON_IMAGES_DIR = '/person_images_data/{datetime}/'
    Recognition.INPUT_IMAGES_DIR = '/images_data/{datetime}/'
    Recognition.ACTIONS_PATH = ''
    Recognition.CNN_MODEL_PATH  = ''
    Recognition.LRCN_MODEL_PATH = ''
    Recognition.PERSON_LRCN_MODEL_PATH = ''
    # params
    # instance
    recognition_instance = Recognition()
    recognition_instance.main()
