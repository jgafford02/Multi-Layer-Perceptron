function [Y] = activation(X,activation_function)
% Description: This function applies the activation function element-wise
% to X
%
% INPUTS:
% X: nodal inputs [nxm matrix]
% activation_function: the desired activation function [string literal]
%
% OUTPUTS:
% Y: the activated nodal outputs [nxm matrix]

% Pre-allocate output
Y = zeros(size(X));

% Update weights depending on desired activation function
switch activation_function
    case 'tanh'
        Y = tanh(X+1)./2;
    case 'logistic'
        Y = 1./(1+exp(-X));
    case 'relu'
        Y = max(zeros(size(X)),X);
    case 'softmax'
        Y = exp(X)./repmat(sum(exp(X),2),1,size(X,2));
end   
end
