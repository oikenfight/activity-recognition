import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz
import cupy


class CnnActivityRecognitionModel(chainer.Chain):
    PRETRAINED_MODEL_PATH = './src/VGG_ILSVRC_19_layers.npz'

    def __init__(self, class_size, pretrained_model=PRETRAINED_MODEL_PATH):
        super(CnnActivityRecognitionModel, self).__init__()
        with self.init_scope():
            self.base = BaseVGG19()
            # self.fc6 = L.Linear(512 * 7 * 7, 4096)
            # self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, class_size)
        npz.load_npz(pretrained_model, self.base)

    def __call__(self, x):
        h = self.base(x)
        # h = F.dropout(F.relu(self.fc6(h)))
        # h = F.dropout(F.relu(self.fc7(h)))
        return self.fc8(h)


class BaseVGG19(chainer.Chain):
    def __init__(self):
        super(BaseVGG19, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.conv2_1 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, 1, 1)
            self.conv3_1 = L.Convolution2D(128, 256, 3, 1, 1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv3_4 = L.Convolution2D(256, 256, 3, 1, 1)
            self.conv4_1 = L.Convolution2D(256, 512, 3, 1, 1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv4_4 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_1 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, 1, 1)
            self.conv5_4 = L.Convolution2D(512, 512, 3, 1, 1)
            self.fc6 = L.Linear(512 * 7 * 7, 4096)
            self.fc7 = L.Linear(4096, 4096)
            # self.fc8 = L.Linear(4096, 1000)

    def __call__(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))

        # これを入れることで、これ以前の層の誤差逆伝搬の計算をしなくなる！
        # CNN ならバッチサイズ確保できるし、全層学習させてもいいけど、LRCN と揃えるため。
        h.unchain_backward()

        h = F.relu(self.conv5_3(h))
        h = F.relu(self.conv5_4(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        # h = self.fc8(h)

        return h
