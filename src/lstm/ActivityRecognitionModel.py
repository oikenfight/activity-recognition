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

    def __call__(self, xs):
        """
        :param xs: transposed feature_batch, it is np or cp array, shape is (frame_size, batch_size, feature_num)
        :return:
        """
        self.lstm.reset_state()
        for x in xs:
            h1 = self.image_vec(x)
            h2 = self.lstm(h1)
        return self.output(h2)


