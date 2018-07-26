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
xtrain = iris.data.astype(np.float32)


class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l1=L.Linear(4, 2),
            l2=L.Linear(2, 4),
        )

    def __call__(self, x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)  # 教師信号が入力データである x になっている

    def fwd(self, x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv


model = MyAE()
optimizer = optimizers.SGD()
optimizer.setup(model)

n = 150
bs = 30
for j in range(3000):
    sffindex = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindex[i:(i+bs) if (i+bs) < n else n]])
        model.cleargrads()
        loss = model(x)
        loss.backward()
        optimizer.update()

#
# Result
#
x = Variable(xtrain)
yt = F.sigmoid(model.l1(x))
ans = yt.data
for i in range(n):
    print(ans[i,:])
