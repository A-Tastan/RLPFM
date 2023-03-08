%% Delta-separated sets
% This function checks whether the it is possible to separate sets which
% are delta=log^(-2/3)(n) separated.
% For details see
%
% [1]A. Taştan, M. Muma and A. M. Zoubir, “Robust Spectral Clustering: A 
% Locality Preserving Feature Mapping Based on M-estimation," in Proc. 29th 
% European Signal Process. Conf., pp. 851-855, 2021.
%
% [2] S. Arora, S. Rao and U. Varizani, “Expander flows, geometric
% embeddings and graph partitioning” J. ACM, vol. 56, pp. 1-37, 2009.
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
% Inputs          :
%                 y_cand          : Estimated candidate Fiedler vector of
%                                   size n x 1
%                 min_num_samples : a scalar contains minimum number of
%                                   samples in per set
%                 decision rule   : a scalar contains decision rule for
%                                   partitioning sets; 1 for cutting using
%                                   zero, 2 for cutting using the median
%                                   (default is 1)
%
% Outputs         :
%                 score_delta     : a real value contains score of delta
%                                   separation based on number of removed
%                                   elements from the sets
%                 gap_initial     : a real value contains initial gap
%                                   between sets
%
% Version      : March 8, 2023
% Author       : Aylin Tastan
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function [score_delta,gap_initial] = check_delta_separated_sets(y_cand,min_num_samples,decision_rule)

if nargin < 3 || isempty(decision_rule)
    decision_rule = 1;
end

%% Create set S and T which are scaled between zero and one
ordered_y_cand = sort(y_cand);
scaled_y_cand = normalize(ordered_y_cand,'range');
if (decision_rule == 1)
    set_S = scaled_y_cand(ordered_y_cand <= 0);
    set_T = scaled_y_cand(ordered_y_cand > 0);
else
    set_S = scaled_y_cand(scaled_y_cand <= median(scaled_y_cand));
    set_T = scaled_y_cand(scaled_y_cand > median(scaled_y_cand));
end

%% Calculate initial gap values
if(~((length(set_S) < min_num_samples)|(length(set_T) < min_num_samples)))
    gap_initial = (norm(min(set_T) - max(set_S)))^2;
else
    gap_initial = 0;
end

%% Make sets S and T delta separated for a minimum information loss
delta = (log10(length(y_cand)))^(-2/3); %delta value
N_removed = 0; %number of elements which are removed
score_delta = 0;
while(~((length(set_S)<min_num_samples)|(length(set_T)<min_num_samples)))
    [max_S,ind_max_S] = max(set_S);
    [min_T,ind_min_T] = min(set_T);
    Gap_S_T = (norm(min_T - max_S))^2;
    if ~(Gap_S_T > delta) %If not delta seperated start remove closer elements
        set_S(ind_max_S) = [];
        set_T(ind_min_T) = [];
        N_removed = N_removed + 2;
    else
        score_delta = 1 -(N_removed / length(y_cand));
        break;
    end
end


end

