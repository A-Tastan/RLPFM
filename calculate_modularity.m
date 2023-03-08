%Calculates the modularity of partition with respect to affinity
%For details, see:
% [1] A. Clauset, M. E. J. Newman, and C. Moore, "Finding community
%     structure in very large networks," Physc. Rev. E., vol.70,pp.066111,
%     2004.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function mod = calculate_modularity(c_hat,W)

m = sum(sum(W));
mod = 0;
cluster = unique(c_hat);
for j=1:length(cluster)
    cluster_j = find(c_hat==cluster(j));
    Ec = sum(sum(W(cluster_j,cluster_j)));
    Et = sum(sum(W(cluster_j,:)));
    if Et>0
        mod = mod + Ec/m-(Et/m)^2;
    end
end

end
