import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets
from chainer.training import extensions
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
X = X.astype(np.float32)
Y = iris.target
Y = Y.flatten().astype(np.int32)

train, test = datasets.split_dataset_random(chainer.datasets.TupleDataset(X, Y), 100)
train_iter = chainer.iterators.SerialIterator(train, 10)
test_iter = chainer.iterators.SerialIterator(test, 1, repeat=False, shuffle=False)


class IrisModel(chainer.Chain):
    def __init__(self):
        super(IrisModel, self).__init__(
                l1=L.Linear(4, 100),
                l2=L.Linear(100, 100),
                l3=L.Linear(100, 3))

    def __call__(self, x):
         h = F.relu(self.l1(x))
         h = F.relu(self.l2(h))
         return self.l3(h)


model = L.Classifier(IrisModel())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (30, 'epoch'), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()
