# Multi-Layer-Perceptron

This repository contains all of the MATLAB files needed to train a DNN. The main function for training the DNN is train_DNN(), which requires an input dataset, and has a number of optional parameters for designing the network.

## Getting Started

train_DNN(dataset,'numNodes',60,'numLayers',3,'rate',0.001,...
    'minibatch',64,'activation','tanh','outputactivation', 'relu',...
    'annealing','none','momentum','adam','regularization','none')

```
data.input_count: a [1x1] scalar containing number of input features (should be m)
data.output_count: a [1x1] scalar containging number of output classes (should be k)
data.training_count: a [1x1] scalar containing number of data points in training set (should be n_train)
data.test_count: a [1x1] scalar containing number of data points in test set (should be n_test)
data.validation_count: a [1x1] scalar containing number of data points in validation set
data.training.input: a [nxm] array containing inputs of the training set (m features, n data points)
data.training.output: a [nxk] array containing outputs of the training set (one-hot vectorized, n data points, k features)
data.training.classes: a [nx1] array containing output classes of the training set (non-vectorized, n data points)
data.test.input: a [nxm] array containing inputs of the test set (m features, n data points)
data.test.output: a [nxk] array containing outputs of the test set (one-hot vectorized, n data points, k features)
data.test.classes: a [nx1] array containing output classes of the test set (non-vectorized, n data points)
data.validation.input: a [nxm] array containing inputs of the validation set (m features, n data points)
data.validation.output: a [nxk] array containing outputs of the validation set (one-hot vectorized, n data points, k features)
data.validation.classes: a [nx1] array containing output classes of the validation set (non-vectorized, n data points)
```

The MLP can be initialized using the following syntax:

train_DNN(dataset,'numNodes',60,'numLayers',3,'rate',0.001,...
    'minibatch',64,'activation','tanh','outputactivation', 'relu',...
    'annealing','none','momentum','adam','regularization','none')
