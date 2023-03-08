%% This demo file runs Robust Spectral Clustering Method based on Robust 
%% Locality Preserving Feature Mapping (RLPFM) method 
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
% 
% Output:
% C_hat                : (numeric) label vector of size n x 1
%
% Version      : March 8, 2023
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

close all;
clear all;
warning off;


%% User inputs
gamma_min = 1e-8;     %Minimum value of gamma
gamma_max = 1000;     %Maximum value of gamma
N_gamma = 10000;      %Sample number for candidate penalties
K_min=1;           %minimum number of clusters
K_max=10;          %maximum number of clusters
sim_measure = 'cosine'; %preferred similarity measure (default : 'cosine')

for MC = 1:100 %100 Monte Carlo runs
    
%% Call Data Set
[X,num_features,num_samples_total,num_clusters] = call_data_set();

%% Define Parameters
gamma_cand = gamma_min:((gamma_max-gamma_min)/(N_gamma)):gamma_max; %candidate penalty parameters
spec_num_clusters=K_min:K_max;                                  %Spesified candidate cluster numbers
min_num_samples=num_samples_total/max(spec_num_clusters);   %min number of samples in per cluster

%% Perform Robust Spectral Clustering
C_hat = perform_robust_spectral_clustering(X, num_samples_total,gamma_cand,min_num_samples,num_clusters,sim_measure);

end


