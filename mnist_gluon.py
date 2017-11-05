#! /usr/bin/python3

import logging

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

def get_lenet():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(10))
    return net

def main():
    batch_size = 100
    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

    # create a trainable module on CPU 0
    lenet_model = get_lenet()

    lenet_model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(lenet_model.collect_params(), 'sgd', {'learning_rate': 0.1})


    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    # train with the same
    epochs = 1
    smoothing_constant = .01

    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())
            with autograd.record():
                output = lenet_model(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0)) else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        test_accuracy = evaluate_accuracy(test_data, lenet_model)
        train_accuracy = evaluate_accuracy(train_data, lenet_model)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
