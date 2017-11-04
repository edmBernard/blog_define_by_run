Idées de titres : 
- Deeplearning Define-by-run neural network (for dummies)
- Deeplearning Define-by-run framework (for dummies)
- Deeplearning Dynamic Neural Network (for dummies)

# Define by run (Dynamic network)

As a new deeplearning user why imperative framework.
New user don't always understand advantage of this kind of framework. In comparison between imperative framework (Chainer, Pytorch, Gluon) and symbolic framework (Tensorflow, Keras, Mxnet ...), Define-and-run framework seem to have bigger community, more activities, more examples ...
In this article, I will separate Mxnet (the symbolic Mxnet API) and Gluon (the imperative Mxnet API).

At the beginning of my journey in deeplearning field (come years ago) and like many new user, I don't see advantage of imperative framework like chainer. And on there website the explaination was not really clear, the speak about easier debugging and other arguments but it doesn't really speak to me. 
In this article, I will try to explain a bit more some of my point of view on why define-by-run framework can be better than define-and-run. That will not be a fair comparison, I will just speak about pros of this approch. 

Who I am, I'm a quite new user in deeplearning. I begin to work with deeplearning alone with self learning. My journey begin with tensorflow at a time where Tensorflow API was awfull and before integration with Keras. My application field is mainly restricted to computer vision so will not present advantage on other field like Speach, Text Recognition or reinforcement learning. I leave Tensorflow to Mxnet, at this time Mxnet have a better support for embed device and a cleaner API. I stay on Mxnet since I use it for my job and on production.

I continue to look imperative framework by curiosity. With the release of Gluon API (Mxnet) a month ago and the pre release of Eager API (Tensorflow) this week, The hype and attention exploded. 
With the release of Gluon, I see a commentaries on why again a new API why don't they finish Keras v2 compliance. I realize the differences between define-by-run and define-and-run API was not known by everyone and new user.

I will try to put the light on some points why there is so much hype on imperative framework/API.

## Debugging

Easier to debug neural network is the main point use in comparison between imperative and symbolic framework. But for new user, when we test 
debugging on MNIST we don't understand the advantage of debugging. This aspect in my opinion only appear with complexe structure or when the issu was not straigth forward. For example, if your network don't converge it's easier to inspect gradient, it's easier to check output dimension.
These issues don't really block on small network more these issues don't appear if you use existing working network. As an end user, these issue never appear as you don't try to adjust layer structure change hidden layer number of feature etc ...

example mxnet vs gluon

## Flexibility

easy to combine network
more flexibility
easier to create complexe architecture (network in concurence)
Shared layer triplet loss network specification

### Less hidden things

This point is more a API design than an advantage of imperative framework. But as most of them (PyTorch, Gluon) take inspiration of Chainer design they show the same level of abstraction.

as imperative the network do what you want and what you tell it.
not inside compilation exemple shared weight.

### Easier to learn

The API design between Chainer, Pytorch and Gluon are really similar this design try to be similar to numpy API. For example, Chainer has develop CuPy (that is now in a separate reposity). CuPy is mainly a clone of Numpy but with the support for operations on GPU.
So in my opinion this framework are more framework agnostic than Tensorflow, Keras or Mxnet. It's easy to switch between them, there is still behaviour variation. And more It's easy to learn because as a scientific python user you already know numpy API.
More in network design it's possible to use regular python loop, condition etc... 

As a example for framework agnostic I will show you 
exemple between pytorch and gluon for shared weight


