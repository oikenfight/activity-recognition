import numpy as np
import pickle as pkl
import chainer
from chainer import cuda, serializers, functions as F
import time
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from ActivityRecognitionModel import ActivityRecognitionModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from FileManager import FileManager


class Classify:
    # constants
    GPU_DEVICE = 0
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512

    def __init__(self, model_path: str, actions_pkl_path: str):
        self.xp = np
        self.actions = self._set_actions(actions_pkl_path)      # {index:action_name} 辞書
        self.model = self._set_model(model_path)
        self._to_gpu()

    def _set_actions(self, path: str):
        self._print_title('Framed Data Loading ...')
        with open(path, 'rb') as f:
            actions = pkl.load(f)
        print('action length: ', str(len(actions)))
        print(actions)
        return actions

    def _set_model(self, path: str) -> ActivityRecognitionModel:
        self._print_title('set model: %s' % path)
        model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, len(self.actions))
        serializers.load_hdf5(path, model)
        model.to_gpu()
        return model

    def _to_gpu(self):
        if self.GPU_DEVICE >= 0:
            self._print_title('use gpu')
            self.xp = cuda.cupy
            self.model.to_gpu(self.GPU_DEVICE)

    def main(self, frame_data: np.ndarray) -> tuple:
        self._print_title('main method.')
        print(self.xp.asarray(frame_data).shape)
        for frame in frame_data:
            # ActivityRecognitionModel で batch_size を取得する部分があるため、1段階無駄にネストしてサイズを shape を統一
            frame = [frame]
            # gpu を使っていれば、cupy に、使ってなければ numpyに変換。されに、0軸 と 1軸を入れ替えて転置
            x = self.xp.asarray(frame).astype(np.float32).transpose((1, 0, 2))
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    y = self.model(x)
                    output = F.softmax(y).data
                    max_value, max_index = float(output.max()), int(output.argmax())
            print('>>> result')
            print(output)
            print('max_index:', str(max_index), 'max:', str(max_value))
            self._predict_action(max_index)
            return max_index, max_value

    def _predict_action(self, max_index: int):
        self._print_title('result action')
        result_action = self.actions[max_index]
        print('>>>> result action is', result_action)

    @staticmethod
    def _print_title(string: str):
        print()
        print("<<< Classify Class: %s >>>" % string)


if __name__ == '__main__':
    #
    # 学習に使ったデータを使って試す。
    #
    model_path = './output/model/20181004_060624/0060.model'
    actions_path = './output/model/20181004_060624/actions.pkl'

    # params
    framed_cnn_pkl_path = './output/framed_cnn/20180929_075743.pkl'
    with open(framed_cnn_pkl_path, 'rb') as f:
        dataset = pkl.load(f)
    print('sentences number: ', str(len(dataset['label'])))
    actions = dataset['actions_dict']
    features = dataset['features']
    labels = dataset['label']

    # 正解率
    acc = 0

    # execute
    classify = Classify(model_path, actions_path)
    for i, (frame, label) in enumerate(zip(features, labels)):
        frame_data = np.array([frame])
        max_index, max_value = classify.main(frame_data)
        if max_index == label:
            acc += 1
        print('結果: %s, 正解: %s, 確率: %s' % (max_index, label, max_value))
        print('正解率:' + str(acc/(i+1)))
        time.sleep(0.5)




