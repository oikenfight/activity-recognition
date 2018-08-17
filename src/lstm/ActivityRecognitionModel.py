import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


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

    def __call__(self, data, label):
        loss = None
        label = Variable(label)
        self.lstm.reset_state()
        print('================================')
        print(data.shape)
        print(data.dtype)
        print('================================')
        for i in range(len(data)):
            print('~~~~~ loop', str(i), '~~~~~~~~~~~~~~~~~')
            for j in range(len(data[i])):
                print('<<<<< loop', str(j), '>>>>>')
                print(len(data[0][0]))
                print(data[0][0])
                h1 = self.image_vec(data[i][j])
                h2 = self.lstm(h1)
                h3 = self.output(h2)

            print(h3)
            if i == len(data) - 1:
                loss = F.softmax_cross_entropy(h3, label)
                accuracy = F.accuracy(h3, label)

        return loss, accuracy


