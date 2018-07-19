function gmresexample(m)
% GMRESEXAMPLE  Compare errors from NAIVEGMRES and built-in GMRES on an  m x m
% tridiagonal example.

b = randn(m,1);
A = diag(1+rand(m,1)) + diag(-rand(1,m-2),-2) + diag(rand(1,m-1),-1) + triu(rand(m,m),1);
x = A \ b;

A

for k = 1:m
    ern(k) = norm(x - naivegmres(A,b,k));
    erb(k) = norm(x - gmres(A,b,k,1.0e-17,1));  % restart = k and maxit = 1
end

figure(1)
subplot(2,2,[1 2])
semilogy(1:m,ern,'ro:;naivegmres;','markersize',14)
hold on
semilogy(1:m,erb,'bx:;built-in gmres;')
set(gca,'XTick',1:m),  grid on
hold off

subplot(2,2,3)
z = eig(A);
plot(real(z),imag(z),'k.'),  grid on
axis([-3 3 -3 3])
title('eigenvalues of A')

subplot(2,2,4)
spy(A)
title('sparsity of A')

