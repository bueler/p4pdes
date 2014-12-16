% GETAMINUS  Used to write c7riemann, to create needed d x d matrices A, A^-

c = 3;
A = [0 c; c 0];
[R,D] = eig(A);
lambda = diag(D);
lambda(lambda>0) = 0;
Aminus = R * diag(lambda) * inv(R)

