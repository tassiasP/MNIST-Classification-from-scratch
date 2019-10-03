% DEMO OF MULTI-CLASS CLASSIFICATION USING A NEURAL NETWORK IN THE MNIST DATASET

clear all; 
close all; 

% Load the MNIST dataset and 
% create the appropritate input and output data matrices 
load mnist_all.mat;
% number of classes
K = 10;
T = []; 
X = [];
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
figure; 
hold on; 
for j=1:10
% 
    s = ['train' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);   
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1; 
    X = [X; Xtmp]; 
    T = [T; Ttmp]; 
    
    s = ['test' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp]; 
   
    % plot some training data
    ind = randperm(size(Xtmp,1));
    for i=1:10
        subplot(10,10,10*(j-1)+i);     
        imagesc(reshape(Xtmp(ind(i),:),28,28)');
        axis off;
        colormap('gray');     
    end
%    
end

% normalzie the pixels to take values in [0,1]
X = X/255; 
Xtest = Xtest/255; 

[N, D] = size(X);

% Add 1 as the first for both the training input and test inputs 
X = [ones(sum(Ntrain),1), X];
Xtest = [ones(sum(Ntest),1), Xtest]; 


M = 200; % The number of hidden units
% Initial W for the gradient ascent

W1init = 0.1*randn(M,D+1);
W2init = 0.1*randn(K,M+1);

% Regularization parameter lambda 
lambda = 0.1; 

% Maximum number of iterations of the gradient ascend
options(1) = 300; 
% Tolerance 
options(2) = 1e-6; 
% Learning rate 
options(3) = 0.5/N;   


% Do a gradient check first
% (in a small random subset of the data so that 
% the gradient check will be fast)

W1 = randn(size(W1init)); 
W2 = randn(size(W2init)); 

ch = randperm(N); 
ch = ch(1:20);
gradcheck_softmax(W1,W2,X(ch,:),T(ch,:),lambda); 


% Train the model 
[W1, W2] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options); 

% Test the model 
[Ttest, Ytest]  = ml_softmaxTest(W1, W2, Xtest); 

[~, Ttrue] = max(TtestTrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;
disp(['The error of the method is: ' num2str(err)])


