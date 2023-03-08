% This function generates an affinity matrix of size n x n
%
% For details, see:
%
% [1]A. Taştan, M. Muma and A. M. Zoubir, “Robust Spectral Clustering: A 
% Locality Preserving Feature Mapping Based on M-estimation," in Proc. 29th 
% European Signal Process. Conf., pp. 851-855, 2021.
%
%
% Copyright (C) 2023 Aylin Tastan. All rights reserved.
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% Inputs:
%       X                 : (numeric) data matrix of size m x n
%                           (m: num_features, n: num_samples_total)
%       num_samples_total : a scalar contains the total number of samples
%       sim_measure       : a string for preferred similarity measure
%                           'cosine','Pearson' or 'enet'
%                           (default is 'cosine')
%       lambda            : (optional) penalty value for elastic net
%                           similarity (default is 0.5)
% Output:
%       W                 : (a numeric) affinity matrix of size n x n
%
% Version      : March 8, 2023
% Dependencies : enet.
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function W = compute_affinity_matrix(X,num_samples_total,sim_measure,lambda)

if(isequal(sim_measure ,'cosine'))

    for i = 1:num_samples_total
        for j = 1:num_samples_total

            W(i,j) = (dot(X(:,i),X(:,j))) / (norm(X(:,i))*norm(X(:,j))); %Cosine similarity
        end
    end
else if(isequal(sim_measure ,'Pearson'))

        W(i,j) = corr(X.'); %Pearson's linear correlation coefficient

else if(isequal(sim_measure ,'enet'))

        if nargin < 4 || isempty(lambda)
            lambda = 0.5;
        end

        alpha = 1 - lambda;
        for j = 1:num_samples_total
            x_j = X(:,j);
            b = X \ x_j;
            [sol,iter] = enet(x_j,X,b,lambda,alpha);
            W(:,j) = sol;
        end
else
    fprintf('Please select a valid similariy measure: i.e. cosine, Pearson or enet');
end
end
end

%Symmetric and zero diagonal affinity matrix
W =(W + W.') / 2;       %symmetric
W = W - diag(diag(W)); %zero diagonal

end