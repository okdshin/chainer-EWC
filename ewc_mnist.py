import os
import argparse
import datetime
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
                    report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter


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
    compute_accuracy = True

    def __init__(self, predictor, lam, num_samples):
        super(EWC, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            self.variable_list = make_variable_list(self.predictor)
        self.fisher_list = None
        self.stored_variable_list = None
        self.lam = lam
        self.num_samples = num_samples
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = F.softmax_cross_entropy(self.y, t)

        if self.stored_variable_list is not None and \
                self.fisher_list is not None:  # i.e. Stored
            for i in range(len(self.variable_list)):
                self.loss += self.lam/2. * F.sum(
                        self.fisher_list[i] *
                        F.square(self.variable_list[i][1] -
                                 self.stored_variable_list[i]))
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = F.accuracy(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
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


def train_task(args, train_name, model, epoch_num,
               train_dataset, test_dataset_dict, batch_size):
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    train_iter = iterators.SerialIterator(train_dataset, batch_size)
    test_iter_dict = {name: iterators.SerialIterator(
            test_dataset, batch_size, repeat=False, shuffle=False)
            for name, test_dataset in test_dataset_dict.items()}

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (epoch_num, 'epoch'), out=args.out)
    for name, test_iter in test_iter_dict.items():
        trainer.extend(extensions.Evaluator(test_iter, model), name)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss'] +
        [test+'/main/loss' for test in test_dataset_dict.keys()] +
        ['main/accuracy'] +
        [test+'/main/accuracy' for test in test_dataset_dict.keys()]))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(
        [test+"/main/accuracy" for test
         in test_dataset_dict.keys()],
        file_name=train_name+".png"))
    trainer.run()


def permutate_mnist(dataset_list):
    indices = np.arange(dataset_list[0][0][0].shape[0])
    np.random.shuffle(indices)

    def permutate_mnist_aux(dataset):
        def transform(in_data):
            img, label = in_data
            return (img[indices], label)
        return datasets.TransformDataset(dataset, transform)
    return (permutate_mnist_aux(dataset) for dataset in dataset_list)


def train_tasks_continuosly(
        args, model, train, test, train2, test2, enable_ewc):
    # Train Task A or load trained model
    if os.path.exists("mlp_taskA.model") or args.skip_taskA:
        print("load taskA model")
        serializers.load_npz("./model50/mlp_taskA.model", model)
    else:
        print("train taskA")
        train_task(args, "train_task_a"+("_with_ewc" if enable_ewc else ""),
                   model, args.epoch, train,
                   {"TaskA": test}, args.batchsize)
        print("save the model")
        serializers.save_npz("mlp_taskA.model", model)

    if enable_ewc:
        print("enable EWC")
        model.compute_fisher(train)
        model.store_variables()

    # Train Task B
    print("train taskB")
    train_task(args, "train_task_ab"+("_with_ewc" if enable_ewc else ""),
               model, args.epoch, train2,
               {"TaskA": test, "TaskB": test2}, args.batchsize)
    print("save the model")
    serializers.save_npz(
            "mlp_taskAB"+("_with_ewc" if enable_ewc else "")+".model", model)


def main():
    output_dim = 10

    parser = argparse.ArgumentParser(description='EWC MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=800,
                        help='Number of sweeps over the dataset to train')
    """
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    """
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--lam', '-l', type=float, default=15.,
                        help='lambda parameter for EWC loss')
    parser.add_argument('--num_samples', '-n', type=int, default=200,
                        help='number of samples to compute fisher')
    parser.add_argument('--hidden', '-hi', type=int, default=50,
                        help='number of hidden node')
    parser.add_argument('--skip_taskA', '-s', type=bool, default=False,
                        help='whether skip training taskA or not')
    args = parser.parse_args()

    model = EWC(MLP(args.hidden, output_dim), args.lam, args.num_samples)

    train, test = chainer.datasets.get_mnist()
    train2, test2 = permutate_mnist([train, test])

    print("Train without EWC")
    train_tasks_continuosly(args, model, train, test, train2, test2,
                            enable_ewc=False)

    print("Train with EWC")
    train_tasks_continuosly(args, model, train, test, train2, test2,
                            enable_ewc=True)


if __name__ == "__main__":
    main()
