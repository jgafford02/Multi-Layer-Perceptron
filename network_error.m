function [error,c,classes] = network_error(data,w,L,activations)
% Description: This function computes the network error (RMS and class
% error)
%
% INPUTS:
% data: Input data [nxm matrix]
% w: cell array containing network weights
% L: vector containing number of nodes in each layer [1xn_L matrix]
% activation_function: string containing the activation function
%
% OUTPUTS:
% error: RMS error between prediction and output 
% c: classification error between prediction and output
% classes: predicted classes

n_L = length(L);
X = data.input;
n_data = size(X,1);
X = horzcat(X, ones(n_data,1));

% Step through layers, pre-allocating space for network activations
a = cell(n_L,1);
for i=1:n_L-1
    a{i} = ones(L(i)+1,1);
end
a{end} = ones(L(end),1);

% Step through layers, pre-allocating space for network inputs
net = cell(n_L-1,1);
for i=1:n_L-2;
    net{i} = ones(L(i+1)+1,1);
end
net{end} = ones(L(end),1);

a{1} = X;

% compute the inputs and outputs to each layer
for i=1:n_L-1
    % compute inputs to this layer
    net{i} = (w{i} * a{i}')';
    
    % compute outputs of this layer
    a{i+1} = activation([net{i}],activations{i});

end
% compute error
[~,output_count] = size(data.output);
error = sum(sum((a{end} - data.output) .^2)) / (data.count * output_count);

% Convert outputs (one-hot vectors) to classes (scalar)
[~,classes] = max(a{end}, [], 2);

% classification error
c = sum(classes ~= data.classes)/data.count;

end

