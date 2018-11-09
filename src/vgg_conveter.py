from chainer.links.caffe.caffe_function import CaffeFunction
from chainer.serializers import npz

caffemodel = CaffeFunction('./src/VGG_ILSVRC_19_layers.caffemodel')
npz.save_npz('VGG_ILSVRC_19_layers.npz', caffemodel, compression=False)
