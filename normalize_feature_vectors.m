% This function computes the normalized feature vectors ||x||_2
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
% Inputs     :
%            X                 :(numeric) data matrix of size m x n
%                               (m:number of features, n:number of samples)
%            num_samples_total : number of observations/samples
%
% Output     :
%            X                 : (numeric) data matrix of size m x n whose
%                                columns are normalized (m:number of
%                                features , n:number of samples)
% Version      : March 8, 2023
% Dependencies : twonorm.
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function X = normalize_feature_vectors(X,num_samples_total)

for j = 1:num_samples_total
    X(:,j) = X(:,j) ./ two_norm(X(:,j));
end

end