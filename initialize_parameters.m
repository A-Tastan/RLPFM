% This function computes initial estimate of eigenvectors based on the
% standard eigen-decomposition.
%
% For details, see:
%
% [1]A. Taştan, M. Muma and A. M. Zoubir, “Robust Spectral Clustering: A 
% Locality Preserving Feature Mapping Based on M-estimation," in Proc. 29th 
% European Signal Process. Conf., pp. 851-855, 2021.
%
% [2] M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral
% techniques for embedding and clustering,” in Proc. Conf. Adv. Neural
% Inf. Process. Syst., vol. 14, 2001.
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
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
%
% Inputs     :
%            W            : (a numeric) affinity matrix of size n x n
%            X            : (numeric) data matrix of size m x n
%                           (m: num_features, n: num_samples_total)
%            num_clusters : (scalar) the number of clusters
%
% Outputs    :
%            y            : eigenvector matrix of size n x k
%            beta_0       : Initial regression matrix of size m x k
%
% Version      : March 8, 2023
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function [y,beta_0] = initialize_parameters(W,X,num_clusters)

%% Computing eigenvectors
D = diag(sum(W)); %Diagonal weight matrix of size n x n

L = D - W;          %Laplacian matrix

[eigenvector_mat,diag_eigenvalue_mat] = eig(L); %eigen-decomposition
[sorted_eigenvalues,ind_sorted_eigenvalues] = sort(diag(diag_eigenvalue_mat));
y = eigenvector_mat(:,ind_sorted_eigenvalues(1:num_clusters)); %eigenvectors associated with k smallest eigenvalues

%% Initial estimate of the transformation vectors
beta_0 = X.' \ y;       %solves the linear equation y=X'beta_0

end
