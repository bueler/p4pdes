function arnoldiblobs(m,L,q)
% ARNOLDIBLOBS  Generate most of Figure 34.3 in Trefethen & Bau, which is on the
% front cover.  "Arnoldi lemniscate" not shown.
% Matrix size is  m,  generates  L x L  subplots, and upper left
% figure shows  n = q + 1  result.  The figure in Tref & Bau is
%     >> arnoldiblobs(100,2,4)

A = randn(m,m) / sqrt(m);  A(1,1) = 1.5;
[Q,H] = arnoldi(A,randn(m,1),q+L^2);
z = eig(A);
for k=1:L^2
    n = q+k;
    y = eig(H(1:n,1:n));
    subplot(L,L,k),  plot(real(y),imag(y),'ro')
    hold on,  plot(real(z),imag(z),'k.')
    text(1,-0.8,sprintf('n = %d',n)),  hold off
    axis equal,  axis off
end

