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
        X(X>0)=1;
        X(X<=0)=0;
        dY = X;
    case 'softmax'
        P = activation(X,'softmax');
        for i = 1:length(X)
            for j = 1:length(P)
                if i==j
                    dY(i)=P(i)*(1-P(i));
                else
                    dY(i) = P(i)*P(j);
                end
            end
        end
        
end    
end

 

