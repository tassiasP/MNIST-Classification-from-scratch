function [Ttest, Ytest]  = ml_softmaxTest(W1, W2, Xtest) 
%function [Ttest, Ytest]  = ml_softmaxTest(W, Xtest) 
%  
% Inputs: 
%         W1, W2: matrices of the parameters   
%         Xtest: Ntest x (D+1) input test data with ones already added in the first column 
% Outputs: 
%         Ttest:  Ntest x 1 vector of the predicted class labels
%         Ytest: Ntest x K matrix of the sigmoid probabilities     
%
% Michalis Titsias (2014)

z = cos(Xtest*W1'); % the activation function for the hidden layer

z = [ones(size(Xtest, 1), 1) , z]; % add the bias column
Ytest = softmax(z*W2');

% Hard classification decisions 
[~,Ttest] = max(Ytest,[],2);