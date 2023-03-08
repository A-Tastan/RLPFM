function [X, num_features, num_samples_total, num_clusters] = call_data_set()

load fisheriris

X = meas.';
[num_features,num_samples_total] = size(X);
num_clusters = 3;

end
