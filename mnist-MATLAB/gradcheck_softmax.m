function gradcheck_softmax(W1,W2,X,T,lambda) 
%
%

[M,D] = size(W1);
[K,M2] = size(W2);  %M2 = M+1 because of the bias

% Compute the analytic gradient 
[~, gradEw1, gradEw2] = costgrad_softmax(W1, W2, X, T, lambda);


% Ccan all parameters to compute 
% numerical gradient estimates
epsilon = 1e-6; 
numgradEw1 = zeros(M,D); 

for k=1:M
    for d=1:D
        Wtmp = W1; 
        Wtmp(k,d) = Wtmp(k,d) + epsilon; 
        Ewplus = costgrad_softmax(Wtmp, W2, X, T, lambda); 
        
        Wtmp = W1; 
        Wtmp(k,d) = Wtmp(k,d) - epsilon; 
        Ewminus = costgrad_softmax(Wtmp, W2, X, T, lambda);
        
        numgradEw1(k,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end

% Display the absolute norm as an indication of how close 
% the numerical gradients are to the analytic gradients
diff = abs(gradEw1 - numgradEw1);  
disp(['The maximum absolute norm in the gradcheck for the gradEw1 is ' num2str(max(diff(:))) ]);



% respectively for gradEw2
numgradEw2 = zeros(K,M2); 
for k=1:K
    for d=1:M2
        Wtmp = W2; 
        Wtmp(k,d) = Wtmp(k,d) + epsilon; 
        Ewplus = costgrad_softmax(W1, Wtmp, X, T, lambda); 
        
        Wtmp = W2; 
        Wtmp(k,d) = Wtmp(k,d) - epsilon; 
        Ewminus = costgrad_softmax(W1, Wtmp, X, T, lambda);
        
        numgradEw2(k,d) = (Ewplus - Ewminus)/(2*epsilon);
    end
end

diff = abs(gradEw2 - numgradEw2);  
disp(['The maximum absolute norm in the gradcheck for the gradEw2 is ' num2str(max(diff(:))) ]);

