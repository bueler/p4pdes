% DECOMPOSEK  Assumes K and np are defined.  Extracts N, nu, A, B.

N = size(K,1);
nu = N - np;
A = K(1:nu,1:nu);
B = K(nu+1:N,1:nu);
K2 = [A B'; B zeros(np,np)];
if norm(K-K2,1) > 0.0
    disp('WARNING: blocks of K not as expected')
end

