import chainer
import chainer.functions as F
import chainer.links as L


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
            h1 = self.image_vec(F.dropout(x, ratio=self.dropout_ratio))
            h2 = self.lstm(F.dropout(h1, ratio=self.dropout_ratio))
        return self.output(F.dropout(h2, ratio=self.dropout_ratio))


