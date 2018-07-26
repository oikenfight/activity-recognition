#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# from chainer.functions import caffe
from chainer.links import caffe
from PIL import Image

from chainer.links import GoogleNet

chainer.using_config('train', False)

#image = Image.open('mydog.jpg').convert('RGB')
image = Image.open('cat2.png').convert('RGB')
fixed_w, fixed_h = 224, 224
w, h = image.size
if w > h:
        shape = (fixed_w * w / h, fixed_h)
else:
        shape = (fixed_w, fixed_h * h / w)

left = (shape[0] - fixed_w) / 2
top = (shape[1] - fixed_h) / 2
right = left + fixed_w
bottom = top + fixed_h
image = image.resize(shape)
image = image.crop((left, top, right, bottom))
x_data = np.asarray(image).astype(np.float32)
x_data = x_data.transpose(2,0,1)  
x_data = x_data[::-1,:,:]

mean_image = np.zeros(3*224*224).reshape(3, 224, 224).astype(np.float32)
mean_image[0] = 103.0
mean_image[1] = 117.0
mean_image[2] = 123.0

x_data -= mean_image 
x_data = np.array([ x_data ])

x = chainer.Variable(x_data)
func = caffe.CaffeFunction('bvlc_googlenet.caffemodel')
# y, = func(inputs={'data': x}, outputs=['loss3/classifier'], train=False)
y, = func(inputs={'data': x}, outputs=['loss3/classifier'])

prob = F.softmax(y)
labels = open('labels.txt').read().split('\n')
maxid = np.argmax(prob.data[0])
print labels[maxid], prob.data[0,maxid]
