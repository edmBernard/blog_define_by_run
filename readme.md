Idées de titres : 
- Deeplearning Define-by-run neural network (for dummies)
- Deeplearning Define-by-run framework (for dummies)
- Deeplearning Dynamic Neural Network (for dummies)

# Define by run (Dynamic network)

As a new deeplearning user why imperative framework.
New user don't always understand advantage of this kind of framework. 
In comparison between imperative framework (Chainer, Pytorch, Gluon) and symbolic framework (Tensorflow, Keras, Mxnet ...), 
Define-and-run framework seem to have bigger community, more activities, more examples ...
In this article, I will separate Mxnet (the symbolic Mxnet API) and Gluon (the imperative Mxnet API).

At the beginning of my journey in deeplearning field (come years ago) and like many new user, 
I don't see advantage of imperative framework like chainer. 
And on there website the explaination was not really clear, 
the speak about easier debugging and other arguments but it doesn't really speak to me. 
In this article, I will try to explain a bit more some of my point of view on why define-by-run framework can be better than define-and-run. 
That will not be a fair comparison, I will just speak about pros of this approch. 

Who I am, I'm a quite new user in deeplearning. 
I begin to work with deeplearning alone with self learning. 
My journey begin with tensorflow at a time where Tensorflow API was awfull and before integration with Keras. 
My application field is mainly restricted to computer vision so will not present advantage on other field like Speach, Text Recognition or reinforcement learning. 
I leave Tensorflow to Mxnet, at this time Mxnet have a better support for embed device and a cleaner API. 
I stay on Mxnet since I use it for my job and on production.

I continue to look imperative framework by curiosity. 
With the release of Gluon API (Mxnet) a month ago and the pre release of Eager API (Tensorflow) this week, 
the hype and attention exploded. 
With the release of Gluon, I see a commentaries on why again a new API why don't they finish Keras v2 compliance. 
I realize the differences between define-by-run and define-and-run API was not known by everyone and new user.

I will try to put the light on some points why there is so much hype on imperative framework/API.

I will show you a full example on three framework Mxnet, Gluon and Pytorch. 
Full code can be found [here]()
I will use them to illustrate my point of view;

If you already now these framework you can directly switch to the end.
So lets begin with an MNIST example :)

Some of these point don't come from imperative particularity but there are de facto for them.
For example, Callable network is possible in Keras, It's one feature that allow an easy network creation.

## Debugging

Easier to debug neural network is the main point use in comparison between imperative and symbolic framework. But for new user, when we test 
debugging on MNIST we don't understand the advantage of debugging. 
This aspect in my opinion only appear with complexe structure or when the issu was not straigth forward. 
For example, if your network don't converge it's easier to inspect gradient, it's easier to check output dimension.
These issues don't really block on small network more these issues don't appear if you use existing working network. 
As an end user, these issue never appear as you don't try to adjust layer structure change hidden layer number of feature etc ...

network can infer input feature dimension.

example mxnet vs gluon
We use the mxnet and gluon with their mid level API to have a better comparison in the syntaxe. 
This is important to note as mxnet and gluon are come from the same framework and the same team Mxnet network definition can be use in Gluon and Gluon network can be freeze to be used with Mxnet and save on disk.

### Network definition

```python
# Mxnet
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
    # full connected layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
    relu3 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=10)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
```

```python
# Gluon
def get_lenet():
    net = gluon.nn.Sequential()
    with net.name_scope():
        # first conv layer
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # second conv layer
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # full connected layer
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(10))
        # softmax loss
        # in Gluon loss functions are separated from network to have a better control
    return net
```

petit paragraphe pour expliquer

### Initialisation and Optimiser definition

On the high level Mxnet API. the fit function hide lot's of things. 
The following "two" line allow to compile network, initialise it, define optimiser, evaluation metric etc...
```python
# Mxnet High level API 
# create a trainable module on CPU 0
lenet_model = mx.mod.Module(symbol=get_lenet(), context=mx.cpu())

# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
```

For a more accurate comparison we will use the slightly low level API for Mxnet.
The relatively low level on Gluon is a design choice to keep control on what you do.

```python
# Mxnet
mod = mx.mod.Module(context=ctx, symbol=m)
mod.bind(data_shapes=[('data', (batch_size, 3, 32, 32))],
            label_shapes=[('softmax_label', (batch_size,))])

mod.init_params(initializer=mx.init.Xavier(rnd_type='uniform'))
mod.init_optimizer(optimizer='sgd', 
                    optimizer_params=(('learning_rate', 0.1), ))
```

```python
# Gluon
lenet_model.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=mx.cpu())

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(lenet_model.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```python
# Mxnet
# Create evaluation metric
metric = mx.metric.create('acc')

# train
for j in range(10):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        model.forward(batch, is_train=True) 
        model.update_metric(metric, batch.label)
        model.backward()              
        model.update()
    print('Epoch %d, Training %s' % (j, metric.get()))
```

```python
# Gluon
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

# train
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

```




```python
# Mxnet
```
```python
# Gluon
```
## Flexibility

Keras do lots of work on their API design to be easy to use. 
Callable network is one the feature that allow an easy network creation. 
And this feature is de facto in define-by-run framework. 
Define-by-run framework add on top of this feature the ability to easily monitoring what happen in these networks.

On symbolic framework there is a network compilation step. 
This step can introduce complexity in network construction. 
For example, if you try to define shared layer, like in siamese network. 
You need to think of where are store your weight, how there are initialized, how there are updated. 
Otherwise, your final network will not be what you attend. 

In imperative framework, you can use regular python condition, loop control flow.
Network with variablility for example pooling with different pooling policy in fonction of input size.
And in my day to day work I don't use RNN, imagine if you have these kind of network to play with this is a huge advantage.

### Less hidden things

This point is more a API design than an advantage of imperative framework. 
But as most of them (PyTorch, Gluon) take inspiration of Chainer design they show the same level of abstraction.

An example on the Mxnet high level API. The fit function is easy to use and do lot's of things. 
But if you want to change things inside iterations, monitor gradient or something else you need to redefine callback, metric, for that you need to now how there work etc ... that was lots of work for sometime just monitoring or debug.
On the high level Mxnet API. the fit function hide lot's of things. 
The following "two" line allow to compile network, initialise it, define optimiser, evaluation metric etc...
```python
# Mxnet High level API 
# create a trainable module on CPU 0
lenet_model = mx.mod.Module(symbol=get_lenet(), context=mx.cpu())

# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
```

as imperative the network do what you want and what you tell it.
not inside compilation exemple shared weight.

### Easier to learn

The API design between Chainer, Pytorch and Gluon are really similar this design try to be similar to numpy API. 
For example, Chainer has develop CuPy (that is now in a separate reposity). 
CuPy is mainly a clone of Numpy but with the support for operations on GPU.
So in my opinion this framework are more framework agnostic than Tensorflow, Keras or Mxnet. 
It's easy to switch between them, there is still behaviour variation. 
And more It's easy to learn because as a scientific python user you already know numpy API.
More in network design it's possible to use regular python loop, condition etc... 

As a example for framework agnostic I will show you 
exemple between pytorch and gluon for shared weight

# Easier learn (not so sure)
don't need to understand function like the fit function in Mxnet

## less hidden thing
# Less hidden things (in apparence maybe more magic)
du design and network compilation
you understand what you do. Maybe harder to run your first network but easier to extend them
easier to made complexe structure

## More flexibility
The true imperative advantage

example shared weight with fc specialisation







## Conclusion 
In my opinion, the hype on define-by-run framework come from a mixed between a good API design and a better flexibility. 
A good API allow to easily understand what happen even if you don't really know the framework, I place Keras with a really good API too. 

This good API combine with a better control on what happen in training loop and what happen in network allow.
So yeah there is so much hype on imperative framework. 
Gains from this kind of framework are not visible at first glance. 
The better flexibility bring by Chainer, Pytorch or Gluon allow to design complexe network easily, with better debugging and with more control.

And cherry on the cake, Chainer, PyTorch and Gluon have similar API (inspired by Chainer).
If you learn one of them it's simple to understand, test other without effort.

easy to learn hard to master 
a bit harder to learn easy to master

easy to do easy thing
hard to do hard thing

a bit harder to do easy thing
easier to do hard thing

Gluon, pytorch


