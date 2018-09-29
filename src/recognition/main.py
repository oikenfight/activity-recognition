import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import frames
import cnn
import lstm


class Recognition:
    # For Output
    OUTPUT_BASE_DIR = "./output/recognition/"
    OUTPUT_JPG_TMP_DIR = OUTPUT_BASE_DIR + "jpg_tmp/%s"   # 後で %s 部分を target_key に置換してユニークなディレクトリを作る

    # For _convert_mp4_to_jpg method.
    FFMPEG_FPS = 2

    # For _get_features_by_cnn method.
    MODEL_PATH = './src/cnn/VGG_ILSVRC_19_layers.caffemodel'

    # For _frame_cnn_features method.
    FRAME_SIZE = 8
    FRAME_OVERLAP_SIZE = 4

    # For _classify_frame_by_lstm method.
    LSTM_MODEL_PATH = './output/model/20180927_063124/0049.model'
    LSTM_ACTIONS_PKL_PATH = './output/model/20180927_063124/actions.pkl'

    def __init__(self, input_video_path: str, lstm_model_path: str, lstm_actions_pkl_path: str):
        self.input_video_path = input_video_path
        self.lstm_model_path = lstm_model_path
        self.lstm_actions_pkl_path = lstm_actions_pkl_path
        self.target_key = self._set_target_key()
        self.cnn_features = []
        self.framed_cnn_features = np.empty((0, 8, 4096), np.float32)

    @staticmethod
    def _set_target_key():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def main(self):
        self._print_title('main method')
        self._prepare_recognition()     # 処理に必要なディレクトリの作成など、下準備しておく
        self._convert_mp4_to_jpg()      # mp4 を jpg に変換して保存
        self._get_features_by_cnn()     # cnn に掛けて特徴抽出する
        self._frame_cnn_features()      # cnn で得た時系列特徴ベクトルをフレームごとに分割
        self._classify_frame_by_lstm()  # cnn のフレームデータから行動内容を予測する

    def _prepare_recognition(self):
        self._print_title('prepare')
        self.OUTPUT_JPG_TMP_DIR = self.OUTPUT_JPG_TMP_DIR % self.target_key + '/'
        os.makedirs(self.OUTPUT_JPG_TMP_DIR)

    def _convert_mp4_to_jpg(self):
        self._print_title("convert mp4 to jpg, input: %s, output_dir: %s" % (self.input_video_path, self.OUTPUT_JPG_TMP_DIR))
        # params
        input_path = self.input_video_path
        output_dir = self.OUTPUT_JPG_TMP_DIR
        # setup
        frames.converter.Converter.FPS = self.FFMPEG_FPS
        # execute
        converter_instance = frames.converter.Converter()
        converter_instance.main(input_path, output_dir)

    def _get_features_by_cnn(self):
        self._print_title("get features by cnn with %s" % self.OUTPUT_JPG_TMP_DIR)
        # params
        input_dir_path = self.OUTPUT_JPG_TMP_DIR
        # setup
        cnn.cnn.Cnn.MODEL_PATH = self.MODEL_PATH
        cnn.cnn.Cnn.GPU_DEVICE = 0
        # execute
        cnn_instance = cnn.cnn.Cnn()
        for i, cnn_vec in enumerate(cnn_instance.main_with_dir(input_dir_path)):
            self.cnn_features += [cnn_vec]
        print('>>> cnn feature length:', str(len(self.cnn_features)))

    def _frame_cnn_features(self):
        self._print_title("frame cnn features by size:%s" % self.FRAME_SIZE)
        for i in range(0, len(self.cnn_features), self.FRAME_OVERLAP_SIZE):
            frame_data = self.cnn_features[i:i+self.FRAME_SIZE]
            if len(frame_data) < 8:
                break
            self.framed_cnn_features = np.append(self.framed_cnn_features, [frame_data], axis=0)
        print('>>> shape of self.framed_cnn_features:', self.framed_cnn_features.shape)

    def _classify_frame_by_lstm(self):
        self._print_title("classify frame by my lstm model")
        # params
        model_path = self.lstm_model_path
        actions_pkl_path = self.lstm_actions_pkl_path
        # setup
        # execute
        classify_instance = lstm.classify.Classify(self.framed_cnn_features, model_path, actions_pkl_path)
        classify_instance.main()

    @staticmethod
    def _print_title(string: str):
        print()
        print("<<< Recognition Class: %s >>>" % string)


if __name__ == "__main__":
    #
    # Example
    #

    # prepare
    model_dir = './output/model/20180929_091536/'

    # params
    input_video_path = '../data/STAIR-actions/stair_action/throwing_trash/a005-0025C.mp4'
    lstm_model_path = model_dir + '0002.model'
    lstm_actions_pkl_path = model_dir + 'actions.pkl'

    # execute
    recognition = Recognition(input_video_path, lstm_model_path, lstm_actions_pkl_path)
    recognition.main()
