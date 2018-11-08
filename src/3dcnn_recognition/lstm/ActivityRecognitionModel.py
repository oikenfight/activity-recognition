import chainer
import chainer.functions as F
import chainer.links as L


class ActivityRecognitionModel(chainer.Chain):
    dropout_ratio = 0.4

    def __init__(self, out_size, lstm_hidden_size=512, train=True):
        super(ActivityRecognitionModel, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3 = L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7 = L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv8 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv9 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv11 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv13 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv14 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv15 = L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv16 = L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc17 = L.Linear(None, 4096)
            self.fc18 = L.Linear(None, 4096)
            self.fc19 = L.Linear(None, lstm_hidden_size)

            self.lstm = L.LSTM(lstm_hidden_size, lstm_hidden_size)
            self.output = L.Linear(lstm_hidden_size, out_size)

    def __call__(self, xs):
        self.lstm.reset_state()
        for x in xs:
            h = F.relu(self.conv1(x))
            h = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv2(h))), 2, stride=2)

            h = F.relu(self.conv3(h))
            h = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv4(h))), 2, stride=2)

            h = F.relu(self.conv5(h))
            h = F.relu(self.conv6(h))
            h = F.relu(self.conv7(h))
            h = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv8(h))), 2, stride=2)

            h = F.relu(self.conv9(h))
            h = F.relu(self.conv10(h))
            h = F.relu(self.conv11(h))
            h = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv12(h))), 2, stride=2)

            h = F.relu(self.conv13(h))
            h = F.relu(self.conv14(h))
            h = F.relu(self.conv15(h))
            h = F.max_pooling_2d(F.local_response_normalization(
                F.relu(self.conv16(h))), 2, stride=2)

            h = F.dropout(F.relu(self.fc17(h)))
            h = F.dropout(F.relu(self.fc18(h)))
            h = F.dropout(F.relu(self.fc19(h)))

            h = self.lstm(F.dropout(h, ratio=self.dropout_ratio))

        self.output(F.dropout(h, ratio=self.dropout_ratio))
        return h
