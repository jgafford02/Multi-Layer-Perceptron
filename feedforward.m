function [output] = feedforward(net,activation_function)

% Notation taken from: http://neuralnetworksanddeeplearning.com/chap2.html

% Activation
% Matrix solution of the following equation:
% a_j^l = sigma(sum_k w_{jk}^l a_k^{l-1} +b_j^l)
% a_j^l: activation of the jth neuron in the lth layer
% sigma: activation function
% w_{jk}^l: weight between kth neuron in (l-1)th layer to jth neuron in l
% layer
% a_k^{l-1}: activation of kth neuron in (l-1)th layer
% b_j^1: bias of jth neuron in lth layer

% Compute network output
% net = [input, bias]*W;

% Apply activation function element-wise
[output] = activation(net, activation_function);

end

