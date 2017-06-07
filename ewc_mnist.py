import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import dataset, datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from sklearn import preprocessing

class MLP(Chain):
    def __init__(self, hidden_num, out_num):
        super(MLP, self).__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, hidden_num)
            self.fc2 = L.Linear(None, hidden_num)
            self.fc3 = L.Linear(None, out_num)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        y = self.fc3(h)
        return y


def make_variable_list(model):
    variable_list = []
    for child in model.children():
        if isinstance(child, Chain):
            variable_list.extend(make_variable_list(child))
        if isinstance(child, Link):
            variable_list.extend(child.namedparams())
    return variable_list


class EWC(Chain):
    def __init__(self, predictor, lam, num_samples):
        super(EWC, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            self.variable_list = make_variable_list(self.predictor)
        self.fisher_list = None
        self.stored_variable_list = None
        self.lam = lam
        self.num_samples = num_samples

    def __call__(self, x, t):
        self.y = self.predictor(x)
        self.accuracy = F.accuracy(self.y, t)
        self.loss = F.softmax_cross_entropy(self.y, t)

        if self.stored_variable_list is not None and \
                self.fisher_list is not None:  # i.e. Stored
            for i in range(len(self.variable_list)):
                self.loss += self.lam/2. * F.sum(
                        self.fisher_list[i] *
                        F.square(self.variable_list[i][1] -
                                 self.stored_variable_list[i]))

        return self.loss

    def compute_fisher(self, dataset):
        fisher_accum_list = [
                np.zeros(var[1].shape) for var in self.variable_list]

        for _ in range(self.num_samples):
            x, _ = dataset[np.random.randint(len(dataset))]
            y = self.predictor(np.array([x]))
            prob_list = F.softmax(y)[0].data
            class_index = np.random.choice(len(prob_list), p=prob_list)
            loss = F.log_softmax(y)[0, class_index]
            self.cleargrads()
            loss.backward()
            for i in range(len(self.variable_list)):
                fisher_accum_list[i] += np.square(
                        self.variable_list[i][1].grad)

        self.fisher_list = [
                F_accum / self.num_samples for F_accum in fisher_accum_list]
        return self.fisher_list

    def store_variables(self):
        self.stored_variable_list = []
        for var in self.variable_list:
            self.stored_variable_list.append(np.copy(var[1].data))


def train_task(model, epoch_num, train_dataset, test_dataset_list, batch_size):
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    train_iter = iterators.SerialIterator(train_dataset, batch_size)
    test_iter_list = [iterators.SerialIterator(
            test_dataset, batch_size, repeat=False, shuffle=False)
            for test_dataset in test_dataset_list]

    accuracy_sum = 0
    loss_sum = 0
    while train_iter.epoch != epoch_num:
        batch = train_iter.next()
        xs, ts = dataset.convert.concat_examples(batch)
        optimizer.update(model, xs, ts)
        loss_sum += float(model.loss.data) * len(ts.data)
        accuracy_sum += float(model.accuracy.data) * len(ts.data)

        if train_iter.is_new_epoch:
            print("epoch:", train_iter.epoch)
            print("train mean loss: {}, accuracy: {}".format(
                loss_sum / len(train_dataset),
                accuracy_sum / len(train_dataset)))
            loss_sum = 0
            accuracy_sum = 0

            # evaluation
            for test_dataset, test_iter in zip(test_dataset_list, test_iter_list):
                test_accuracy_sum = 0
                test_loss_sum = 0
                for batch in test_iter:
                    xs, ts = dataset.convert.concat_examples(batch)
                    model(xs, ts)
                    test_loss_sum += float(model.loss.data) * len(ts.data)
                    test_accuracy_sum += float(model.accuracy.data) * len(ts.data)
                test_iter.reset()
                print("test mean loss: {}, accuracy: {}".format(
                    test_loss_sum / len(test_dataset),
                    test_accuracy_sum / len(test_dataset)))


def permutate_mnist(dataset_list):
    indices = np.arange(dataset_list[0][0][0].shape[0])
    np.random.shuffle(indices)

    def permutate_mnist_aux(dataset):
        def transform(in_data):
            img, label = in_data
            return (img[indices], label)
        return datasets.TransformDataset(dataset, transform)
    return (permutate_mnist_aux(dataset) for dataset in dataset_list)


def main():
    epoch_num = 800
    batch_size = 128
    hidden_num = 400
    output_dim = 10
    lam = 15
    num_samples = 400
    model = EWC(MLP(hidden_num, output_dim), lam, num_samples)

    # Train Task A
    #train, test = chainer.datasets.get_mnist()
    train, test = chainer.datasets.get_cifar10()
    train_task(model, epoch_num, train, [test], batch_size)
    print("save the model")
    serializers.save_npz("mlp_taskA.model", model)
    #serializers.load_npz("model50/mlp_taskA.model", model)

    model.compute_fisher(train)
    model.store_variables()

    # Train Task B
    train2, test2 = permutate_mnist([train, test])
    train_task(model, epoch_num, train2, [test2, test], batch_size)
    print("save the model")
    serializers.save_npz("mlp_taskAB.model", model)
    #serializers.load_npz("model50/mlp_taskAB.model", model)

    #train_task(model, epoch_num, train2, [test2, test], batch_size)


if __name__ == "__main__":
    main()
