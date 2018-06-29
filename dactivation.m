function [dY] = dactivation(X,activation_function)
% Description: This function applies the derivative of the activation 
%function element-wise to X
%
% INPUTS:
% X: nodal inputs [nxm matrix]
% activation_function: the desired activation function [string literal]
%
% OUTPUTS:
% Y: the derivative of the activated nodal outputs [nxm matrix]

dY = zeros(size(X));

% Update weights depending on desired activation function
switch activation_function
    case 'tanh'
        dY = (1-tanh(X).^2)./2;
    case 'logistic'
        out = 1./(1+exp(-X));
        dY = out.*(1-out);
    case 'relu'
        dY = X>0;
    case 'leakyrelu'
        alpha = 0.01;
        dY = alpha.*(X<0)+(X>0);
    case 'softmax'
        P = activation(X,'softmax');
        dY = P.*(1-P);
        
end    
end

 

