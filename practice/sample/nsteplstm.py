#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

vocab = {}

def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']

class MyLSTM(chainer.Chain):
    def __init__(self, lay, v, k, dout):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(v, k),
            H = L.NStepLSTM(lay, k, k, dout),            
            W = L.Linear(k, v),
        )
    def __call__(self, hx, cx, xs, t):
        accum_loss = None
        xembs = [ self.embed(x) for x in xs ]
        xss = tuple(xembs)
        hy, cy, ys = self.H(hx, cx, xss)
        y = [self.W(item) for item in ys]
        for i in range(len(y)):
            tx = Variable(np.array(t[i], dtype=np.int32))            
            loss = F.softmax_cross_entropy(y[i], tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 100
model = MyLSTM(2, len(vocab),  demb,  0.5)
optimizer = optimizers.Adam()
optimizer.setup(model)

bc = 0
xs = []
t = []
for epoch in range(5):
    s = [] 
    for pos in range(len(train_data)):
        id = train_data[pos]
        if (id != eos_id):
            s += [ id ]
        else:
            bc += 1
            next_s = s[1:]
            next_s += [ eos_id ] 
            xs += [ np.asarray(s, dtype=np.int32) ]
            t += [ np.asarray(next_s, dtype=np.int32) ]
            s = []
            if (bc == 10):
                model.cleargrads()
                hx = chainer.Variable(np.zeros((2, len(xs), demb), dtype=np.float32))
                cx = chainer.Variable(np.zeros((2, len(xs), demb), dtype=np.float32))
                loss = model(hx, cx, xs, t)
                loss.backward()
                optimizer.update()
                xs = []
                t = []
                bc = 0
        if (pos % 100 == 0):
            print pos, "/", len(train_data)," finished"
    outfile = "nsteplstm-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
