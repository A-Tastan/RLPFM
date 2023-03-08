%% Robust Spectral Clustering: A Locality Preserving Feature Mapping 
%% Based on M-estimation
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
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <https://www.gnu.org/licenses/>.
% 
% !NOTE : 
% This code requires an additional function 'whub' which is available in :
%
% https://github.com/RobustSP/toolbox/blob/master/codes/07_AuxiliaryFunctions/whub.m
% 
% Further, it may require "enet.m" and "SoftThresh.m" functions if the
% preferred similarity measure is the elastic net similarity. The codes
% are, respectively, available in :
% 
% https://github.com/RobustSP/toolbox/blob/master/codes/02_Regression/enet.m
%
% and
%
% https://github.com/RobustSP/toolbox/blob/master/codes/07_AuxiliaryFunctions/SoftThresh.m
%
% Lastly, it may require an additional function if the matrix XX in
% "Robust_Regularized_Locality_Preserving_Indexing.m" is rank deficient.
% An example function that is named "RankDeficientFastDecomposition.m" code
% is available in:
% 
% [2] P. Courrieu, “Fast computation of Moore-Penrose inverse
% matrices.” Online-Edition: https://arxiv.org/abs/0804.4809, 2008

% Inputs:
%       X                     : (numeric) data matrix of size m x n
%                                (m: num_features, n: num_samples_total)
%       num_samples_total     : a scalar contains the total number of
%                                samples in the data set
%       gamma_cand            : a vector containing the candidate penalty
%                                parameters
%       min_num_samples       : minimum number of samples in per set for
%                                delta seperated sets
%       num_clusters          : a scalar contains number of clusters
%       sim_measure           : a string for preferred similarity measure,
%                                'cosine','Pearson' or 'enet'
%                               (default is 'cosine')
%       func_opt              : a scalar contains function option(default2)
%                                1-> degree-based regularized locality
%                                preserving indexing, 2-> residual-based
%                                regularized locality preserving indexing
%       decision_rule         : a scalar contains decision rule for delta
%                                seperated sets; 1 for cutting using zero,
%                                2 for cutting using median (default is 1)
%
% Outputs:
%        C_hat                : (numeric) label vector of size n x 1
%
%
% Version      : March 8, 2023
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function C_hat = perform_robust_spectral_clustering(X,num_samples_total,gamma_cand,min_num_samples,num_clusters,sim_measure,func_opt,decision_rule)

if nargin < 8 || isempty(decision_rule)
    decision_rule = 1;
end

if nargin < 7 || isempty(func_opt)
    func_opt = 2;
end

if nargin < 6 || isempty(sim_measure)
    sim_measure = 'cosine';
end

%% Initialization
X = normalize_feature_vectors(X,num_samples_total); %the feature vectors are normalized i.e. ||x||_2
W = compute_affinity_matrix(X,num_samples_total,sim_measure);
[y,beta_0] = initialize_parameters(W,X,num_clusters); %Initial estimate of eigenvectors and transformation vectors

%% Eigenvector Estimation using RLPFM
for i = 1:length(gamma_cand)

    %RLPFM
    [y_cand,~] = robust_locality_preserving_feature_mapping(X,W,y,gamma_cand(i),func_opt,beta_0);

    %Delta Seperated Sets
    [score_delta,gap_initial] = check_delta_separated_sets(y_cand(:,2),min_num_samples,decision_rule);

    %Store Initial gap and Delta scores
    Gap_val(i) = gap_initial;
    Score_delta_val(i) = score_delta;

end

if (any(Score_delta_val)) %If any subsets provide delta seperation, pick the ones with maximum initial gap
    [~,ind_gamma_est] = max(Score_delta_val);
else
    [~,ind_gamma_est] = max(Gap_val);
end

%Compute eigenvectors for the estimated regularization parameter
[y_hat,~] = robust_locality_preserving_feature_mapping(X,W,y,gamma_cand(ind_gamma_est),func_opt,beta_0);

%% Partitioning
[C_hat,~] = kmeans(y_hat,num_clusters,'Replicates',10); %Perform k-means on robustly estimated eigenvectors

end