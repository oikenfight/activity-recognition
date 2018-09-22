import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import cuda
import time


class ActivityRecognitionModel(chainer.Chain):
    dropout_ratio = 0.5
    cp = cuda.cupy

    def __init__(self, feature_size, hidden_size, out_size):
        """
        :param feature_size: 入力層サイズ
        :param hidden_size: 隠れ層サイズ
        :param out_size: 出力層サイズ
        """
        super(ActivityRecognitionModel, self).__init__(
            image_vec=L.Linear(feature_size, hidden_size),
            lstm=L.LSTM(hidden_size, hidden_size),
            output=L.Linear(hidden_size, out_size)
        )

    def __call__(self, data, label):

        data = self.cp.asarray(data).astype(np.float32)
        label = self.cp.asarray(label).astype(np.int32)

        loss = None
        accuracy = None
        self.lstm.reset_state()

        t = label.astype(np.int32)

        # TODO: ignore_part は src/cnn/create_cnn_dataset.py でミスったため、shape が期待と違うため。余計に1階層ネストしてる。
        batch_size, ignore_part, feature_length, feature_size = data.shape
        for col in range(feature_length):
            x = data[:, 0, col]    # TODO: ここも上のと同じ。0列目を指定して、無駄なネストを無視する感じ。
            h1 = self.image_vec(x)
            h2 = self.lstm(h1)
            h3 = self.output(h2)

            if col == feature_length - 1:
                loss = F.softmax_cross_entropy(h3, t)
                accuracy = F.accuracy(h3, t)
        return loss, accuracy


