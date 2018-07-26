#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                        optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

jvocab = {}
jlines = open('jp.txt').read().split('\n')
for i in range(len(jlines)):
    lt = jlines[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)

jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)
            
evocab = {}
id2wd = {}
elines = open('eng.txt').read().split('\n')
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            id = len(evocab)
            evocab[w] = id
            id2wd[id] = w

id = len(evocab)            
evocab['<eos>'] = id
id2wd[id] = '<eos>'
ev = len(evocab)

def mk_ct(gh, ht):
    alp = []
    s = 0.0    
    for i in range(len(gh)):
        s +=  np.exp(ht.dot(gh[i]))
    ct = np.zeros(100)
    for i in range(len(gh)):
        alpi = np.exp(ht.dot(gh[i]))/s
        ct += alpi * gh[i]
    ct = Variable(np.array([ct]).astype(np.float32))
    return ct

class MyATT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyATT, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),  
            H = L.LSTM(k, k),
            Wc1 = L.Linear(k, k),
            Wc2 = L.Linear(k, k),            
            W = L.Linear(k, ev),
        )
    def __call__(self, jline, eline):
        gh = []
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)
            gh.append(np.copy(h.data[0]))
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        ct = mk_ct(gh, h.data[0])
        h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
        accum_loss = F.softmax_cross_entropy(self.W(h2), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))                        
            next_wid = evocab['<eos>']  if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            ct = mk_ct(gh, h.data)
            h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))            
            loss = F.softmax_cross_entropy(self.W(h2), tx)
            accum_loss += loss 
        return accum_loss
    def reset_state(self):
        self.H.reset_state()

demb = 100
model = MyATT(jv, ev, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    for i in range(len(jlines)-1):
        jln = jlines[i].split()
        jlnr = jln[::-1]
        eln = elines[i].split()
        model.reset_state()
        model.zerograds()        
        loss = model(jlnr, eln)
        loss.backward()
        loss.unchain_backward()  # truncate
        optimizer.update()
        print i, " finished"
    outfile = "attention-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)

