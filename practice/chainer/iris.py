#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn import datasets

xp = cuda.cupy  ## added

iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3 * N).reshape(N, 3).astype(np.float32)
for i in range(N):
    Y2[i, Y[i]] = 1.0

index = np.arange(N)
# index が奇数・偶数でtrainデータとtestデータに分割
xtrain = X[index[index % 2 != 0], :]
ytrain = Y2[index[index % 2 != 0], :]
xtest = X[index[index % 2 == 0], :]
yans = Y[index[index % 2 == 0]]

# gpu を使う準備
xtrain = xp.array(xtrain).reshape(75, 4)  # added
ytrain = xp.array(ytrain).reshape(75, 3)  # added
xtest = xp.array(xtest).reshape(75, 4)    # added

#
# Define model
#
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    # 損失関数（入力: x, 出力: y）:2乗誤差
    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    # 順方向の計算
    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


#
# Initialize model
#
model = IrisChain()
cuda.get_device(0).use()  # use gup
model.to_gpu()
# 最適化関数をセット（損失関数のパラメータで微分し、損失を最小化するためその微分値からパラメータを更新する）
optimizer = optimizers.SGD()
optimizer.setup(model)

#
# Learn
#
for i in range(10000):
    x = Variable(xtrain)
    y = Variable(ytrain)
    model.cleargrads()   # 勾配の初期化
    loss = model(x, y)   # 順方向に計算して誤差を算出（__call__ メソッドが呼ばれる）
    loss.backward()      # 逆方向の計算、勾配の計算
    optimizer.update()   # パラメータを更新（一回のバッチで一回パラメータが更新される）

#
# Test
#
xt = Variable(xtest)
yy = model.fwd(xt)
ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i, :])
    print(ans[i, :], cls)
    if cls == yans[i]:
        ok += 1

print(ok, "/", nrow, " = ", (ok * 1.0) / nrow)
