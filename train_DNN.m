function [w] = train_DNN(dataset,data,varargin)
% Description: Feedforward and Backpropagation through multi-level hidden
% neueral network (minibatch)
%
% INPUTS
% dataset: dataset name [string literal]
% varargin: value-pair options for running algorithm
%           {'numNodes', [1x1 scalar]}: Specify number of nodes in hidden layer
%           {'numLayers', [1x1 scalar]}: Specify number of hidden layers
%           {'rate', [1x1 scalar]}: Learning rate
%           {'minibatch', [1x1 scalar]}: Size of minibatch
%           {'annealing', [string]}: rate annealing (possible options:
%               {'none','step','exponential','1/t'}
%           {'activation', [string]}: activation function (possible options:
%               {'tanh','logistic','relu','leakyrelu','softmax'}
%           {'momentum', [string]}: momentum (possible options:
%               {'none','regular','nesterov','adam'}
%           {'regularization', [string]}: regularization (possible options:
%               {'none','dropout','L2','L1'}
%           {'plot_graphs', [boolean]}: Plot error graphs at the end of
%               runtime.
%
% OUTPUTS:
% w: cell structure containing weight matrices

% Written by: Joshua Gafford
% Data: 06/21/2018
% Notation taken from: http://neuralnetworksanddeeplearning.com/chap2.html

%--------------------------------------------------------------------------
% Input handling and default initializations
%--------------------------------------------------------------------------

% Default initializations
defaultNodes = 5;
defaultLayers = 5;
defaultRate = 0.005;
defaultMiniBatch = 16;
defaultMaxEpoch = 20;    % Maximum number of iterations
defaultTargetAccuracy = 90;

% Parse inputs
p = inputParser;
addRequired(p, 'dataset', @ischar);
addOptional(p, 'numNodes', defaultNodes, ...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'numLayers', defaultLayers, ...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'rate', defaultRate, ...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'minibatch', defaultMiniBatch, ...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'annealing', 'none', ...
    @(s) any(validatestring(s, {'none','step','exponential','1/t'})));
addOptional(p, 'activation', 'tanh', ...
    @(s) any(validatestring(s,{'tanh','logistic','relu','leakyrelu','softmax'})));
addOptional(p, 'outputactivation', 'softmax', ...
    @(s) any(validatestring(s,{'tanh','logistic','relu','leakyrelu','softmax'})));
addOptional(p, 'momentum', 'none', ...
    @(s) any(validatestring(s, {'none','regular','nesterov','adam'})));
addOptional(p, 'regularization', 'none', ...
    @(s) any(validatestring(s, {'none','L1','L2','dropout'})));
addOptional(p, 'plotgraphs', true,...
    @(x) isboolean(x));
addOptional(p, 'maxEpochs', defaultMaxEpoch,...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'targetAccuracy', defaultTargetAccuracy,...
    @(x) isnumeric(x) && isscalar(x) && (x>0) && (x<=100));

parse(p,dataset,data,varargin{:});
           
n_n = p.Results.numNodes;
n_h = p.Results.numLayers;
n_L = n_h+2;                    % Number of total layers
n_batch = p.Results.minibatch;
rate = p.Results.rate;
annealer = p.Results.annealing;
activation_function = p.Results.activation;
output_activation_function = p.Results.outputactivation;
momentum = p.Results.momentum;
regularization = p.Results.regularization;
plot_graphs = p.Results.plotgraphs;
max_epoch = p.Results.maxEpochs;
target_accuracy = p.Results.targetAccuracy;

activations = {};
for i=1:n_L-1
    if i==(n_L-1)
        activations{i} = activation_function;
    else
        activations{i} = output_activation_function;
    end
end

data = load_data(strcat(dataset,'.mat'));

% format for data .mat file:
% data: structure containing input data. The struct must have the
% following elements:
%
%           data.input_count: a [1x1] scalar containing number of input
%           features (should be m)
%           data.output_count: a [1x1] scalar containging number of output
%           classes (should be k)
%
%           data.training_count: a [1x1] scalar containing number of data
%           points in training set (should be n_train)
%           data.test_count: a [1x1] scalar containing number of data
%           points in test set (should be n_test)
%           data.validation_count: a [1x1] scalar containing number of data
%           points in validation set
%
%           data.training.input: a [nxm] array containing inputs of the
%           training set (m features, n data points)
%           data.training.output: a [nxk] array containing outputs of the
%           training set (one-hot vectorized, n data points, k features)
%           data.training.classes: a [nx1] array containing output
%           classes of the training set (non-vectorized, n data points)
%
%           data.test.input: a [nxm] array containing inputs of the
%           test set (m features, n data points)
%           data.test.output: a [nxk] array containing outputs of the
%           test set (one-hot vectorized, n data points, k features)
%           data.test.classes: a [nx1] array containing output
%           classes of the test set (non-vectorized, n data points)
%
%           data.validation.input: a [nxm] array containing inputs of the
%           validation set (m features, n data points)
%           data.validation.output: a [nxk] array containing outputs of the
%           validation set (one-hot vectorized, n data points, k features)
%           data.validation.classes: a [nx1] array containing output
%           classes of the validation set (non-vectorized, n data points)

fprintf('---------------Dense Neural Network---------------------\n');
fprintf('Dataset: %s\n',dataset);
fprintf('Number of Nodes: %i   Number of Hidden Layers: %i\n',n_n,n_h);
fprintf('Minibatch size: %i    Learning Rate: %i\n',n_batch,rate);
fprintf('Momentum: %s    Annealer: %s    Regularization:%s\n',momentum, ...
    annealer, regularization);
fprintf('Activation Function: %s\n',activation_function);


n_data = data.training_count;   % Number of samples in training set
acc = 0;                    % pre-allocate training accuracy
max_batch_idx = floor(n_data/n_batch);
max_num_batches = max_batch_idx.*max_epoch;

% Vector containing size of each layer
L = [size(data.training.input,2);n_n.*ones(n_h,1);...
    size(data.training.output,2)];

% Step through layers, pre-allocating space for weight vectors
w = cell(n_L-1,1);
for i=1:n_L-2
    w{i} = [.5 - rand(L(i+1),L(i)+1) ; zeros(1,L(i)+1)];
end
w{end} = .5 - rand(L(end),L(end-1)+1);

% Step through layers, pre-allocating space for neuron activations
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

% Pre-allocate for epoch and error vectors (for max iteration)
epoch = zeros(1,max_num_batches-1);
error_train = epoch;    
c_train = epoch;
error_test = epoch;     
c_test = epoch;
error_validation = epoch;   
c_validation = epoch;
a_test = epoch;  
a_train = epoch;
a_val = epoch;

% Initialize iterator and timer
tic;
iter = 1;

batch_idx = 1;
epoch_idx = 1;
loss = 0;
while ((epoch(batch_idx)<max_epoch)&&(acc<target_accuracy))
    
    % Compute current epoch
    epoch(batch_idx+1) = batch_idx*n_batch/n_data;
    
    % Randomly sample data and create sequential minibatch
    rand_ind = randsample(size(data.training.input,1),1);
    if(rand_ind-n_batch<1)
        samp = 1:1:n_batch;
    elseif ((rand_ind+n_batch)>n_data)
        samp = n_data:-1:(n_data-n_batch+1);
    else
        samp = rand_ind:1:(rand_ind+n_batch-1);
    end

    % Index into input and output data for minibatch
    X = data.training.input(samp,:);    % Sample input layer
    X = horzcat(X, ones(n_batch,1));    % Append 1's to act as bias to input
    Y = data.training.output(samp,:);   % Sample Output layer
    
    % First activation layer is input layer
    a{1} = X;
    
    %----------------------------------------------------------------------
    % Feedforward to get net outputs and activation for all layers
    %----------------------------------------------------------------------
    for i=1:n_L-1 
        % Network output from previous layer
        net{i} = network_output(a{i},w{i},regularization); 
        % Activation for current layer
        a{i+1} = activation([net{i}],activations{i});
    end
    
    % Compute error vector
    batch_loss = 0.5*(sum(sum(Y-a{end},1))).^2;
    loss = loss + batch_loss;
    error_vector = (Y - a{end});
    
    % activation gradients and deltas for each layer
    sigma = dactivation(net{end},activations{end});
    delta = error_vector.*sigma;
    
    % %--------------------------------------------------------------------
    % % Backpropagate to adjust weights in hidden layer
    % %--------------------------------------------------------------------
    annealed_rate = anneal(rate,annealer,epoch(iter),.02);
    for i=n_L-1:-1:1
        dw_mean = delta'*a{i};  % Sum of all delta activations
        w{i} = update_weights(w{i},dw_mean,i,momentum,...
            regularization,annealed_rate,n_L);
        if i > 1
            % Update sum of delta activations for next layer
            sigma = dactivation(net{i-1},activations{i-1});
            delta = (delta*w{i}).*sigma;
        end
    end
    
    % Only compute error/classification metrics after each epoch
    if ~(mod(batch_idx,max_batch_idx))
        % Compute average loss for epoch
        ave_loss = loss/max_batch_idx;
        loss = 0;
        
        % Compute RMSE and classification accuracy for training, testing
        % and validation sets
        [error_train(epoch_idx),c_train(epoch_idx),output_train] = ...
            network_error(data.training,w,L,activations);
        [error_test(epoch_idx),c_test(epoch_idx),output_test] = ...
            network_error(data.test,w,L,activations);
        [error_validation(epoch_idx),c_validation(epoch_idx),output_validation] = ...
            network_error(data.validation,w,L,activations);
        [a_train(epoch_idx),a_test(epoch_idx),a_val(epoch_idx)] = ...
            accuracy(data,output_train,output_test,output_validation);
        
        % Print some results
        fprintf('\n-----------End of Epoch %i------------\n', epoch_idx);
        fprintf('Test Set Accuracy: %f Average Loss: %f ',a_test(epoch_idx),ave_loss);
        fprintf('\n-----------Start of Epoch %i------------\n' ,epoch_idx+1);
        epoch_idx = epoch_idx+1;    % Update epoch index
    end
    
    % Update batch index
    batch_idx = batch_idx+1;
end
total_elapsed_time = toc;

% if algorithm converged before max iteration, resize results to eliminate 
% trailing zeros
epoch = [0 epoch(1:epoch_idx-1)];
error_train = error_train(1:epoch_idx-1);    
c_train = c_train(1:epoch_idx-1);
error_test = error_test(1:epoch_idx-1);     
c_test = c_test(1:epoch_idx-1);
error_validation = error_validation(1:epoch_idx-1);   
c_validation = c_validation(1:epoch_idx-1);
a_test = [0 a_test(1:epoch_idx-1)];  a_train = [0 a_train(1:epoch_idx-1)];
a_val = [0 a_val(1:epoch_idx-1)];

% Plot data if plot_graphs = true
if plot_graphs
    figure(1)
    % RMS error plot
    set(gcf,'Position',[100 100 1200 450]);
    subplot(1,2,1)
    epoch_vec = 0:1:epoch_idx-1;
    plot(epoch_vec(2:end),error_train,'-ro',...
        'MarkerFaceColor','r','MarkerEdgeColor','k');hold on;
    plot(epoch_vec(2:end),error_test,'-gs','MarkerFaceColor',...
        'g','MarkerEdgeColor','k');hold on;
    plot(epoch_vec(2:end),error_validation,'-b^','MarkerFaceColor',...
        'b','MarkerEdgeColor','k');hold on;
    legend('Training','Test','Validation');
    xlabel('Epoch');
    ylabel('RMS Error');
    title(strcat(dataset,' RMS Error (Adaptive: ',momentum,')'));
    
    % Classification Accuracy plot
    subplot(1,2,2)
    plot(epoch_vec,a_train,'-ro','MarkerFaceColor','r',...
        'MarkerEdgeColor','k');hold on;
    plot(epoch_vec,a_test,'-gs','MarkerFaceColor','g',...
        'MarkerEdgeColor','k');hold on;
    plot(epoch_vec,a_val,'-b^','MarkerFaceColor','b',...
        'MarkerEdgeColor','k');hold off;
%     xlim([epoch(1) epoch(end)]);
    legend('Training','Test','Validation','Location','Southeast');
    xlabel('Epoch');
    ylabel('Classification Accuracy [percent]');
    title(strcat(dataset,' Classification Accuracy (Adaptive: ',momentum,')'));
end


% Print results to command
fprintf('Neural Net Results\n');
fprintf('Number of epochs for convergence: %i\n',epoch(end));
fprintf('Total Training Time: %f s\n',total_elapsed_time);
fprintf('Training Classification Success Rate: %f percent\n',...
    100*sum(output_train == data.training.classes)/data.training.count);
fprintf('Testing Classification Success Rate: %f percent\n',...
    100*sum(output_test == data.test.classes)/data.test_count);
fprintf('Validation Classification Success Rate: %f percent\n',...
    100*sum(output_validation == data.validation.classes)/data.validation_count);

% Query user to save plots, metadata, and optimized weights
str = input('Save plot files, metadata, and optimized weights? [y/n]','s');
if isempty(str)
    str = 'y';
end

% If yes, plots to pdfs and metadata to a text file
if strcmp(str,'y')
    figure(1)
    set(gcf,'Renderer','painters');
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',...
        [pos(3), pos(4)])
    filename_string = strcat(dataset,'_multilayerANN_Nh',num2str(n_h),...
        '_Nn',num2str(n_n),'_mb',num2str(n_batch),'_',momentum,'_',annealer);
    print(gcf,strcat('results/',filename_string,'_error'),'-dpdf');

    % save metadata to text file
    fid = fopen(strcat('results/',filename_string,'_stats.txt'), 'wt' );
    fprintf(fid,'Neural Net Results\n');
    fprintf(fid,'Data Set: %s\n',dataset);
    fprintf(fid,'Number of Hidden Layers: %i\n',n_h);
    fprintf(fid,'Number of Neurons per Hidden Layer: %i\n',n_n);
    fprintf(fid,'Size of minibatch: %i\n',n_batch);
    fprintf(fid,'Hidden Activation Function: %s\n',activation_function);
    fprintf(fid,'Output Activation Function: %s\n',output_activation_function);
    fprintf(fid,'Adaptive Parameter Update: %s\n',momentum);
    fprintf(fid,'Number of Input Features: %i\n',data.input_count);
    fprintf(fid,'Number of Output Features/Labels: %i\n',data.output_count);
    fprintf(fid,'Number of Training Instances: %i\n',data.training_count);
    fprintf(fid,'Number of total epochs: %i\n',epoch(end));
    fprintf(fid,'Total Training Time: %f s\n',total_elapsed_time);
    fprintf(fid,'Training RMS Error: %f\n',error_train(end));
    fprintf(fid,'Test RMS Error: %f\n',error_test(end));
    fprintf(fid,'Validation RMS Error: %f\n',error_validation(end));
    fprintf(fid,'Training Classification Success Rate: %f percent\n',...
        100*sum(output_train == data.training.classes)/length(data.training.input));
    fprintf(fid,'Testing Classification Success Rate: %f percent\n',...
        100*sum(output_test == data.test.classes)/data.test_count);
    fprintf(fid,'Validation Classification Success Rate: %f percent\n',...
        100*sum(output_validation == data.validation.classes)/data.validation_count);
    fclose(fid);
    
    % save weights and errors to .mat files
    save(strcat('weights/',filename_string,'_W.mat'),'w',...
        'activations');
    save(strcat('results/',filename_string,'_error.mat'),'error_train',...
        'c_train','error_test','c_test','error_validation','c_validation');
end


end

