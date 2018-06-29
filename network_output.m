function net = network_output(a,w,regularization)
% Description: This function computes the network input to the current
% layer
%
% INPUTS:
% a: activations from previous layer [nx1 matrix]
% w: weighting matrix between previous layer and current layer [nxm]
% regularization: the desired regularization method [string literal]
%
% OUTPUTS:
% net: the network input to the current layer [mx1 matrix]

% Hyperparameter
p = 0.85;

% Modify network output depending on regularization method
switch regularization
    case 'dropout'
        net_ur = (w*a')';   %unregularized network output
        H = (rand(size(net_ur))<p)./p;      % construct dropout mask
        net = net_ur.*H;    %apply mask to unregularized output
    otherwise
        net = (w*a')';
end

