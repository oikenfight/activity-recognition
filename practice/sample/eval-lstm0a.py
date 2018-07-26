#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import collections
import math
import sys
argvs = sys.argv

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

class MyLSTM(chainer.Chain):
    def __init__(self, v, k):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(v, k),
            Wz = L.Linear(k, k),
            Wi = L.Linear(k, k),
            Wf = L.Linear(k, k), 
            Wo = L.Linear(k, k),
            W = L.Linear(k, v),
        )
    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(xp.zeros((1,k), dtype=xp.float32))
        c = Variable(xp.zeros((1,k), dtype=xp.float32)) 
        for i in range(len(s)):
            next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(xp.array([next_w_id], dtype=xp.int32))
            x_k = self.embed(Variable(xp.array([s[i]], dtype=xp.int32)))
            z0 = self.Wz(x_k) + self.Wz(h)
            z1 = F.tanh(z0)
            i0 = self.Wi(x_k) +  self.Wi(h)
            i1 = F.sigmoid(i0)
            f0 = self.Wf(x_k) +  self.Wf(h)
            f1 = F.sigmoid(f0)
            c = i1 * z1 + f1 * c
            o0 = self.Wo(x_k) +  self.Wo(h)
            o1 = F.sigmoid(o0)
            h = o1 * F.tanh(c)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

train_data = load_data('ptb.train.txt')

demb = 100
def cal_ps(model, s):
    sum = 0.0
    h = Variable(np.zeros((1,demb), dtype=np.float32))
    c = Variable(np.zeros((1,demb), dtype=np.float32))
    for i in range(1,len(s)):
        w1, w2 = s[i-1], s[i]    
        x_k = model.embed(Variable(np.array([w1], dtype=np.int32)))
        z0 = model.Wz(x_k) + model.Wz(h)
        z1 = F.tanh(z0)
        i0 = model.Wi(x_k) +  model.Wi(h)
        i1 = F.sigmoid(i0)
        f0 = model.Wf(x_k) +  model.Wf(h)
        f1 = F.sigmoid(f0)
        c = i1 * z1 + f1 * c
        o0 = model.Wo(x_k) +  model.Wo(h)
        o1 = F.sigmoid(o0)
        y = o1 * F.tanh(c)
        yv = F.softmax(model.W(y))
        pi = yv.data[0][w2]
        sum -= math.log(pi, 2)
    return sum
        
eos_id = vocab['<eos>']
max_id = len(vocab)
test_data = load_data('ptb.test.txt')
test_data = test_data[0:1000]
model = MyLSTM(len(vocab), demb)
serializers.load_npz(argvs[1], model)

sum = 0.0
wnum = 0
s = []
unk_word = 0    
for pos in range(len(test_data)):
    id = test_data[pos]
    s.append(id)     
    if (id > max_id):
        unk_word = 1     
    if (id == eos_id):
        if (unk_word != 1):
            ps = cal_ps(model, s)
            sum += ps
            wnum += len(s) - 1
        else:
            unk_word = 0                
        s = []                        
print math.pow(2, sum / wnum)



