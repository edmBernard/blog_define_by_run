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
With the release of Gluon, I see a commentaries on why again a new API why don't they finish Keras v2 compliance. I realize the differnce between define-by-run and define-and-run API was not known by every on and new user.

I will try to put the light on some point why there is so much hype on imperative framework/API.

## noob point of view

debugging
not visible on small network 
example mxnet vs gluon

## modular network

easy to combine network
more flexibility 
easier to create complexe architecture (network in concurence)

## no hidden thing

as imperative the network do what you want and what you tell it. 
not inside compilation exemple shared weight.

that was more framework agnostic 
like normal python/numpy normal for loop if while etc ...
exemple between pytorch and gluon for shared weight


