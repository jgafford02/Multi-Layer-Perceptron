function [annealed_rate] = anneal(rate,annealer,epoch,k)
% Description: This function anneals the learning rate
%
% INPUTS:
% rate: the nominal rate [scalar]
% annealer: the desired annealing function [string literal]
% epoch: the current epoch [scalar]
% k: hyperparameter [scalar]
%
% OUTPUTS:
% annealed_rate: the annealed rate

switch annealer
    case 'exponential'
        annealed_rate = rate*exp(-k*epoch)+rate/20;
    case '1/t'
        annealed_rate = rate/(1+k*epoch)+rate/20;
    case 'none'
        annealed_rate = rate;
end
       
end

