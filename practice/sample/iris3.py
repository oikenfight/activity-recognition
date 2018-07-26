#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

# Set data

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int32)
N = Y.size
index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y[index[index % 2 != 0]]
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

# Define model

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
        
    def __call__(self,x,y):
        return F.softmax_cross_entropy(self.fwd(x), y)        

    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2

# Initialize model

model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

# Learn

n = 75    
bs = 25   
for j in range(2000):   
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = Variable(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        loss = model(x,y)
        loss.backward()
        optimizer.update()

# Test

xt = Variable(xtest)
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print ans[i,:], cls            
    if cls == yans[i]:
        ok += 1
        
print ok, "/", nrow, " = ", (ok * 1.0)/nrow

