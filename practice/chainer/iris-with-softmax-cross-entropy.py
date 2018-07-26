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
Y = iris.target.astype(np.int32)
N = Y.size
index = np.arange(N)
xtrain = X[index[index % 2 != 0], :]
ytrain = Y[index[index % 2 != 0]]   # 教師信号は整数値 (0 or 1 or 2)
xtest = X[index[index % 2 == 0], :]
yans = Y[index[index % 2 == 0]]

#
# Define model
#
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4, 6),
            l2=L.Linear(6, 3),
        )

    # 損失関数（入力: x, 出力: y）: softmax_cross_entropy
    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    # 順方向の計算
    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


#
# Initialize model
#
model = IrisChain()
# 最適化関数をセット（損失関数のパラメータで微分し、損失を最小化するためその微分値からパラメータを更新する）
optimizer = optimizers.SGD()
optimizer.setup(model)

#
# Learn
#
n = 75
bs = 25
for j in range(5000):
    sffindex = np.random.permutation(n)  # 0 ~ n のランダムな配列を生成
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindex[i:(i+bs) if (i+bs) < n else n]])
        y = Variable(ytrain[sffindex[i:(i+bs) if (i+bs) < n else n]])
        model.cleargrads()
        loss = model(x, y)
        loss.backward()
        optimizer.update()

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


