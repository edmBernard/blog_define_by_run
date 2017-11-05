#! /usr/bin/python3

import logging

import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.ndarray as F
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu')
            self.pool1 = gluon.nn.MaxPool2D(pool_size=2, strides=2)
            self.conv2 = gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu')
            self.pool2 = gluon.nn.MaxPool2D(pool_size=2, strides=2)
            self.fc1 = gluon.nn.Dense(512, activation="relu")
            self.fc2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.reshape((0, -1))
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = x.reshape((0, -1))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


def main():
    # Get data
    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)

    batch_size = 100
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

    # Define Network and loss
    lenet_model = Net()
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # Initialize
    lenet_model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())
    trainer = gluon.Trainer(lenet_model.collect_params(), 'sgd', {'learning_rate': 0.1})

    # Create evaluation metric
    # Create evaluation metric don't forget that softmax was not in the model definition
    # For the evaluation we need to add argmax at the network output
    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    # Train
    print("training ...")
    smoothing_constant = .01
    for e in range(10):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            with autograd.record():
                output = lenet_model(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])

            # define a moving loss for log
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        test_accuracy = evaluate_accuracy(test_data, lenet_model)
        train_accuracy = evaluate_accuracy(train_data, lenet_model)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
