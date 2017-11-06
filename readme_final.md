# A sneak peek into Dynamic Neural Networks


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

What exactly is Dynamic Neural network, "Define-by-Run" frameworks, Imperative framework.

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

You can explore your network step by step and monitor what happen to each layer.

> Difficulty in debugging: While static analysis permits some errors to be identified during declaration, many logic errors will necessarily wait to be uncovered until execution (especially when many variables are left underspecified at declaration time), which is necessarily far removed from the declaration code that gave rise to them. This separation of the location of the root cause and location of the observed crash makes for difficult debugging.

* regular debugger
* step by step forward pass

## Model definition

Now almost all framework have userfriendly network definition

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
    return net
```

Keras do lots of work on their API design to be easy to use. 
Callable network is one the feature that allow an easy network creation. 
And this feature is de facto in define-by-run framework. 
> Ok, now we see that it’s easy to build some if/else/while complex statements in PyTorch. But let’s revert to the usual models. The framework provides out of the box layers constructors very similar to Keras ones:
* Models definition
box layers constructors very similar to Keras ones
Define-by-run framework add on top of this feature the ability to easily monitoring what happen in these networks.

> Difficulty in expressing complex flow-control logic: When computing the value of a loss function requires traversal of complex data structures or the execution of nontrivial inference algorithms, it is necessary, in the static declaration paradigm, to express the logic that governs these algorithms as part of the computation graph. However, in existing toolkits, iteration, recursion, and even conditional execution (standard flow control primitives in any high-level language) look very different in the graph than in the imperative coding style of the host language—if they are present at all [43, 8]. Since traversal and inference algorithms can be nontrivial to implement correctly under the best of circumstances [19], implementing them “indirectly” in a computation graph requires considerable sophistication on the developer’s part.

## Flexibility

On symbolic framework  is a network compilation step. 
This step can introduce complexity in network construction. 
For example, if you try to define shared layer, like in siamese network. 
You need to think of where are store your weight, how there are initialized, how there are updated. 
Otherwise, your final network will not be what you attend. 

In imperative framework, you can use regular python condition, loop control flow.
Network with variablility for example pooling with different pooling policy in fonction of input size.
And in my day to day work I don't use RNN, imagine if you have these kind of network to play with this is a huge advantage.

A small example to illustrate that.
Begin with this statement, you want to extract attribut from face. You want to extract gender and glass. Is it a male or a female and Is it wearing glass.
But your biggest problem is you don't have database with both label. You only have a database with gender and one with glass.
One of the way to solve this is to create a root network shared between both and define specialize layer for each funcionnality.
This method reduce the weight to have two full network in your prediction flow.


## Expressiveness


This point is more a API design than an advantage of imperative framework. 
But as most of them (PyTorch, Gluon) take inspiration of Chainer design they show the same level of abstraction.

An example on the Mxnet high level API. The fit function is easy to use and do lot's of things. 
But if you want to change things inside iterations, monitor gradient or something else you need to redefine callback, metric, for that you need to now how there work etc ... that was lots of work for sometime just monitoring or debug.
On the high level Mxnet API. the fit function hide lot's of things. 
The following "two" line allow to compile network, initialise it, define optimiser, evaluation metric etc...


```python
# Mxnet High level API 
# Train
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)
```
Full code example can be found [here]()
as imperative the network do what you want and what you tell it.
not inside compilation exemple shared weight.


no compilation less hidden operations
More regular python 
less framework unique function

## Easier to learn

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


More natural flow

## Conclusion

In my opinion, the hype on define-by-run framework come from a mixed between a good API design and a better flexibility. 
A good API allow to easily understand what happen even if you don't really know the framework, I place Keras with a really good API too. 

If you take one of the easiest API on symbolic framework (Keras), we can describe it as "easy to learn hard to master". 
Whereas, Define-by-run frameworks are a bit harder to learn but they allow complexe network creation with serenity.


This good API combine with a better control on what happen in training loop and what happen in network allow.
So yeah there is so much hype on imperative framework. 
Gains from this kind of framework are not visible at first glance. 
The better flexibility bring by Chainer, Pytorch or Gluon allow to design complexe network easily, with better debugging and with more control.

Dynamic framework : 
- can be use as drop-in replacement of Numpy
- They are really fast for prototyping
- It’s easy to debug and use conditional flows

Define-by-run framework are fast-growing framework. And I think this time is the good time to try them out!
And cherry on the cake, Chainer, PyTorch and Gluon have similar API (inspired by Chainer).
If you learn one of them it's simple to understand, test other without effort.


