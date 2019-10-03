function [W1, W2] = ml_softmaxTrain(T, X, lambda, W1init, W2init, options)
%function W = ml_softmaxTrain(t, X, lambda, Winit, options)
%
% What it does: It trains using gradient ascend a linear logistic regression  
%               model with regularization
%
% Inputs: 
%         T: N x K binary output data matrix indicating the classes
%         X: N x (D+1) input data vector with ones already added in the first column
%         lambda: the positive regularizarion parameter
%         W1init: M x (D+1) matrix of the initial values of the parameters
%         W2init: K x (M+1) matrix of the initial values of the parameters
%         options: options(1) is the maximum number of iterations 
%                  options(2) is the tolerance
%                  options(3) is the learning rate eta 
% Outputs: 
%         W1, W2: the trained matrices of the parameters     
%  
% Michalis Titsias (2016)

W1 = W1init;
W2 = W2init;

% Maximum number of iteration of gradient ascend
iter = options(1); 

% Tolerance
tol = options(2);

% Learning rate
eta = options(3);
 
Ewold = -Inf; 
for it=1:iter
%    
    % Call the cost function to compute both the value of the cost
    % and its gradients. You should store the value of the cost to 
    % the variable Ew and the gradients to a K x (D+1) matrix gradEw
    [Ew, gradEw1, gradEw2] = costgrad_softmax(W1, W2, X, T, lambda);
    
    % Show the current cost function on screen
    fprintf('Iteration: %d, Cost function: %f\n',it, Ew); 

    % Break if you achieve the desired accuracy in the cost function
    if abs(Ew - Ewold) < tol 
        break;
    end
    
    % Update parameters based on gradient ascent 
    W1 = W1 + eta*gradEw1;
    W2 = W2 + eta*gradEw2;
    
    Ewold = Ew; 
%
end
        