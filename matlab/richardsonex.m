% RICHARDSONEX  the example of preconditioned Richardson iteration in Chapter 1

A = [10 -1; -1 1];
P = [10 0; 0 1];
uexact = [1; 2];
b = A * uexact;

M = zeros(2,5);
z = [0; 0];
for k = 1:4
  z = z + 1.0 * (b - A*z);
  M(:,k+1) = z;
end
M

M = zeros(2,5);
z = [0; 0];
for k = 1:4
  z = z + 1.0 * (P \ b - P \ A*z);
  M(:,k+1) = z;
end
M

eig(eye(2) - A)

eig(eye(2) - P \ A)
