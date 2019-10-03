function [Ew, gradEw1, gradEw2] = costgrad_softmax(W1, W2, X, T, lambda)

a1 = X*W1';
Z = cos(a1); % activation function
Z = [ones(size(Z,1),1), Z]; %bias(like in the X matrix in the demo file)
Yx = Z*W2'; %equal to a2

K = size(W2, 1);
    
% Compute the cost in a super-numerically stable way 
M = max(Yx, [], 2); %it takes the max across rows of the matrix Yx
% it applies the logsumexp trick 
Ew = sum(sum( T.*Yx )) - sum(M)  - sum(log(sum(exp(Yx - repmat(M, 1, K)), 2)))  - (0.5*lambda)*sum(sum(W2.*W2));
  
% Return also the gradients if neeeded
if nargout > 1  
   % softmax probabilities 
   S = softmax(Yx);
   % gradientEw2
   gradEw2 = ((T - S)')*Z - lambda*W2;
   
   %gradientEw1
   W2 = W2(:, 2:end);
    
   gradEw1 = (W2'*(T-S)') * X - lambda*W1;
end