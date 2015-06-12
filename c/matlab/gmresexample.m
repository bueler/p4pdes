% GMRESEXAMPLE

m = 20;  b = randn(m,1);
A = eye(m) + diag(-rand(1,m-1),1) + diag(rand(1,m-1),-1) + diag(rand(1,m-2),-2);
x = A\b;

for k=1:m
    ern(k) = norm(x - naivegmres(A,b,k));
    erb(k) = norm(x - gmres(A,b,k));
end

semilogy(1:m,ern,'o:')
hold on
semilogy(1:m,erb,'*:')
set(gca,'XTick',1:m),  grid on

