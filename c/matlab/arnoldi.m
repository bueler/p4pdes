function [Q, H] = arnoldi(A,b,n)
% ARNOLDI  If  A  is  m x m  and  b  is  m x 1  and  n < m  then
%     [Q,H] = arnoldi(A,b,n)
% returns  m x (n+1)  matrix  Q  which is orthogonal  (Q'*Q = I_n)
% and  (n+1) x n  matrix  H  so that
%     A * Q(:,1:n) = Q(:,1:n+1) * H
% that is,
%     A [q_1|...|q_n] = [q_1|...|q_n+1] H
% and where  q_1 = b / |b|_2.  If  n == m  then  Q, H  are the same size as
% A and  A * Q = Q * H.
%
% See Algorithm 33.1 in Trefethen & Bau.
%
% Example:  basic small matrix example
%     >> A = 0.3 * randn(6,6) + eye(6);
%     >> [Q,H] = arnoldi(A,randn(6,1),4);
%     >> eig(A), eig(H(1:4,:))
%
% Example:  reproduce figure 34.3 (=front cover) from Trefethen and Bau
%     >> arnoldiblobs(100,2,4)

m = size(A,1);
if size(A,2) ~= m, error('A must be square'), end

if n < m
    Q = zeros(m,n+1);  H = zeros(n+1,n);
elseif n == m
    Q = zeros(m,m);  H = zeros(m,m);
else
    error('n <= m is required, where m is dim of A')
end

Q(:,1) = b / norm(b);

for k = 1:n
    v = A * Q(:,k);
    for j = 1:k
        H(j,k) = Q(:,j)' * v;
        v = v - H(j,k) * Q(:,j);
    end
    if k == m, break; end
    H(k+1,k) = norm(v);
    if H(k+1,k) == 0, error('break-down at step %d',k), end
    Q(:,k+1) = v / H(k+1,k);
end

