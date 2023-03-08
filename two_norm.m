% twonorm: Computes ||v||_2
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
function f = two_norm(v)

f = sqrt(sum(abs(v(:)).^2));

end
    

