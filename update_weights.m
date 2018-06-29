function w_update = update_weights(w_in,delta_w,layer,momentum,regularization,rate,n_L)
% Description: This function applies momentum or per-parameter adaptation
% based on the algorithm specified in 'momentum'
%
% INPUTS:
% w_in: the weigthing matrix of the current iteration [nxm matrix]
% delta_w: the gradient [nxm matrix]
% layer: the current layer in the DNN [1x1 scalar]
% momentum: the desired algoirthm [string literal]
% rate: the learning rate [1x1 scalar]
% n_L: the total number of layers in the network [1x1 scalar]
%
% OUTPUTS:
% w_updata: the updated weighting matrix [nxm matrix]

% Persistent variables to preserve states between function calls
persistent v;
persistent m;
persistent v_prev;
persistent init;

% momentum hyperparameters
mu = 0.5;           % For regular/Nesterov momentum
beta1 = 0.9;        % For ADAM
beta2 = 0.99;       % For ADAM
eps = 1E-8;         % For ADAM

% If first initialization, allocate space 
if ~strcmp(momentum,'none')&&isempty(init)
    v = cell(n_L);
    v_prev = cell(n_L);
    m = cell(n_L);
    init = 1;
end

% If first time accessing current layer, initialize place in cell
if ~strcmp(momentum,'none')&&isempty(v{layer})
    v{layer} = zeros(size(delta_w));
    v_prev{layer} = zeros(size(delta_w));
    m{layer} = zeros(size(delta_w));
end

% Update weights depending on desired momentum method
switch momentum
    case 'regular'
        v{layer} = -mu.*v{layer}+rate.*delta_w;
        w_update = w_in + v{layer};
    case 'nesterov'
        v_prev{layer} = v{layer};
        v{layer} = -mu*v{layer} + rate.*delta_w;
        w_update = w_in + mu*v_prev{layer} + (1+mu).*v{layer};
    case 'adam'
        m{layer} = beta1.*m{layer} + (1-beta1).*delta_w;
        v{layer} = beta2.*v{layer} + (1-beta2).*(delta_w.^2);
        w_update = w_in + rate.*m{layer}./(sqrt(v{layer})+eps);
    case 'none'
        w_update = w_in + rate.*delta_w;
end

lambda = 0.00001;

% Modify gradient update depending on regularization method
switch regularization
    case 'L2'
        w_update = w_update+lambda.*w_in;
    case 'L1'
        w_update = w_update+lambda.*sign(w_in);
    otherwise
        w_update = w_update;
end
end

