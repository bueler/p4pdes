function x = naivegmres(A,b,n)
% NAIVEGMRES  is Algorithm 35.1 from Trefethen & Bau, but using QR to solve
% the least-squares problem
% FIXME: example

% get H for "small" minimization problem
[Q,H] = arnoldi(A,b,n);

% solve "small" minimization problem
r = [norm(b); zeros(n-1,1)];  % right-hand-side
[QH,RH] = qr(H,'0');          % economy-size
y = RH \ (QH' * r);

% recover approx solution
x = Q(:,1:n) * y;

