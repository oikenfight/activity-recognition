import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from Dataset import Dataset
from ActivityRecognitionModel import ActivityRecognitionModel


class Train:
    EPOCH_NUM = 1000
    FEATURE_SIZE = 4096
    HIDDEN_SIZE = 512
    BACH_SIZE = 30
    TEST_RATE = 0.25

    def __init__(self):
        self.xp = np
        self.dataset = self.set_dataset()
        self.train_data, self.test_data = self.dataset.shuffle_to_train_test(self.TEST_RATE)
        self.model = self.set_model()
        self.optimizer = self.set_optimizer()

    def to_gpu(self, gpu_id: int):
        self._print_title('use gpu')
        cuda.check_cuda_available()
        cuda.get_device(gpu_id).use()
        self.xp = cuda.cupy

    def set_dataset(self) -> Dataset:
        self._print_title('set dataset')
        dataset = Dataset()
        return dataset

    def set_model(self) -> ActivityRecognitionModel:
        self._print_title('set model:')
        model = ActivityRecognitionModel(self.FEATURE_SIZE, self.HIDDEN_SIZE, self.dataset.get_output_size())
        return model

    def set_optimizer(self):
        self._print_title('set optimizer')
        optimizer = optimizers.Adam()
        optimizer.setup(self.model)
        return optimizer

    def main_with_iterator(self):
        self._print_title('main with iterator ')
        train_data = tuple_dataset.TupleDataset(self.train_data[0], self.train_data[1])
        train_iter = iterators.SerialIterator(train_data, self.BACH_SIZE)
        updater = training.StandardUpdater(train_iter, self.optimizer)
        trainer = training.Trainer(updater, (self.EPOCH_NUM, 'epoch'))
        trainer.extend(extensions.ProgressBar())
        trainer.run()

    @staticmethod
    def _print_title(s: str):
        """
        現在の処理内容とデバッグしたいデータ（配列）を渡す）
        :param str s:
        :return:
        """
        print()
        print('<<<<', s, '>>>>')


if __name__ == '__main__':
    train = Train()
    # train.to_gpu(0)
    train.main_with_iterator()
