#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy  ## added

# Set data

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

index = np.arange(N)
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0],:]
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

xtrain = xp.array(xtrain).reshape(75,4)  ## added
ytrain = xp.array(ytrain).reshape(75,3)  ## added 
xtest = xp.array(xtest).reshape(75,4)    ## added 

# Define model

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
        
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2

# Initialize model

model = IrisChain()
cuda.get_device(0).use()       ## added 
model.to_gpu()                 ## added 
optimizer = optimizers.SGD()
optimizer.setup(model)

# Learn
for i in range(10000):
    x = Variable(xtrain)
    y = Variable(ytrain)
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

