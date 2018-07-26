#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
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

class MyGRU(chainer.Chain):
    def __init__(self, v, k):
        super(MyGRU, self).__init__(
            embed = L.EmbedID(v, k),
            H  = L.StatefulGRU(k, k),            
            W = L.Linear(k, v),
        )
    def __call__(self, s):
        accum_loss = None
        v, k = self.embed.W.data.shape
        self.H.reset_state()                 
        for i in range(len(s)):
            next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            tx = Variable(np.array([next_w_id], dtype=np.int32))
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32)))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 100    
model = MyGRU(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(5):
    s = []    
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)        
        if (id == eos_id):
            loss = model(s)
            model.cleargrads()
            loss.backward()
            if (len(s) > 29):
                loss.unchain_backward()  # truncate                         
            optimizer.update()
            loss = 0            
            s = []                
        if (pos % 100 == 0):
            print pos, "/", len(train_data)," finished"
    outfile = "gru-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
