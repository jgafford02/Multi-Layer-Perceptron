# Multi-Layer-Perceptron

This repository contains all of the MATLAB files needed to train a simple DNN. The main function for training the DNN is train_DNN(), which requires an input dataset, and has a number of optional parameters for designing the network.

## Getting Started

The training session can be initialized by calling the function:

train_DNN(dataset,'numNodes',60,'numLayers',3,'rate',0.001,...
    'minibatch',64,'activation','tanh','outputactivation', 'relu',...
    'annealing','none','momentum','adam','regularization','none')
    
Note that it is assumed that the string 'dataset' refers to a .mat file consisting of a struct with the following entries:

```
input_count: a [1x1] scalar containing number of input features (m features)
output_count: a [1x1] scalar containging number of output classes (k classes)
training_count: a [1x1] scalar containing number of data points in training set (n_train)
test_count: a [1x1] scalar containing number of data points in test set (n_test)
validation_count: a [1x1] scalar containing number of data points in validation set (n_val)
training.input: a [n_train x m] array containing inputs of the training set (m features, n_train data points)
training.output: a [n_train x k] array containing outputs of the training set (one-hot vectorized, n_train data points, k features)
training.classes: a [n_train x 1 ] array containing output classes of the training set (n_train data points)
test.input: a [n_test x m] array containing inputs of the test set (m features, n_test data points)
test.output: a [n_test x k] array containing outputs of the test set (one-hot vectorized, n_test data points, k features)
test.classes: a [n_test x 1] array containing output classes of the test set (non-vectorized, n_test data points)
validation.input: a [n_val x m] array containing inputs of the validation set (m features, n_val data points)
validation.output: a [n_val x k] array containing outputs of the validation set (one-hot vectorized, n_val data points, k features)
validation.classes: a [n_val x 1] array containing output classes of the validation set (non-vectorized, n_val data points)
```
I have included iris.mat and wdbc.mat which are already in this form.

## Version
This code was developed on MATLAB 2016a.

## Author
Written by Joshua Gafford
