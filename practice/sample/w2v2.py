#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer.utils import walker_alias
import collections

# Set data

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []
with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])

n_vocab = len(word2index)
datasize = len(dataset)

cs = [counts[w] for w in range(len(counts))]
power = np.float32(0.75)
p = np.array(cs, power.dtype)
sampler = walker_alias.WalkerAlias(p)

# Define model

class MyW2V2(chainer.Chain):
    def __init__(self, v, m):
        super(MyW2V2, self).__init__(
            embed = L.EmbedID(v,m),
        )
    def __call__(self, xb, eb, sampler, ngs):
        loss = None        
        for i in range(len(xb)):
            x = Variable(np.array([xb[i]], dtype=np.int32))
            e = eb[i]
            ls = F.negative_sampling(e, x, self.embed.W, sampler, ngs)
            loss = ls if loss is None else loss + ls            
        return loss

# my functions
    
ws = 3    ### window size
def mkbatset(model, dataset, ids):
    xb, eb = [], []
    for pos in ids:    
        xid = dataset[pos]
        for i in range(1,ws):
            p = pos - i
            if p >= 0:
                xb.append(xid)
                eid = dataset[p]
                eidv = Variable(np.array([eid], dtype=np.int32)) 
                ev = model.embed(eidv)
                eb.append(ev)
            p = pos + i
            if p < datasize:
                xb.append(xid)
                eid = dataset[p]
                eidv = Variable(np.array([eid], dtype=np.int32))
                ev = model.embed(eidv)                
                eb.append(ev)
    return [xb, eb] 

# Initialize model

model = MyW2V2(n_vocab, 100)
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learn

bs = 50
ngs = 5

for epoch in range(10):
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(datasize)
    for pos in range(0, datasize, bs):    
        print epoch, pos
        ids = indexes[pos:(pos+bs) if (pos+bs) < datasize else datasize]
        xb, eb = mkbatset(model, dataset, ids)
        model.cleargrads()        
        loss = model(xb, eb, sampler.sample, ngs)
        loss.backward()
        optimizer.update()

# Save model        

with open('w2v2.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), 100))
    w = model.lf.W.data
    for i in range(w.shape[0]):
        v = ' '.join(['%f' % v for v in w[i]])
        f.write('%s %s\n' % (index2word[i], v))



    
