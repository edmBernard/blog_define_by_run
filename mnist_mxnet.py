#! /usr/bin/python3

import logging

import mxnet as mx

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

def get_lenet():
    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
    relu3 = mx.sym.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=10)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def main():
    mnist = mx.test_utils.get_mnist()    
    batch_size = 100
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    mod = mx.mod.Module(context=ctx, symbol=m)
    mod.bind(data_shapes=[('data', (batch_size, 3, 32, 32))], label_shapes=[('softmax_label', (batch_size,))])

    mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))

if __name__ == '__main__':
    main()
