function x = naivegmres(A,b,n)
% NAIVEGMRES  is Algorithm 35.1 from Trefethen & Bau, using QR to solve
% the "small" least-squares problem.  Less naive versions would save work by
% using Givens rotations to perform QR as the Arnoldi iteration proceeds.
% Calls ARNOLDI.
% Example:
%     >> gmresexample(20)

if size(A,1) ~= size(A,2), error('A must be square'), end
if size(b,1) ~= size(A,1) | size(b,2) ~= 1, error('b must be column vector for A'), end
n = min(n,size(A,1));

% get H for "small" minimization problem
[QA,H] = arnoldi(A,b,n);      % H is n+1 x n

% solve "small" minimization problem
if n == size(A,1),  c = zeros(n,1);  else,  c = zeros(n+1,1);  end
c(1) = norm(b);        % right-hand-side:  c = |b| e_1
[Q,R] = qr(H,'0');     % economy-size:  Q is size of H, R is n x n
y = R \ (Q' * c);

% recover approx solution using Q
x = QA(:,1:n) * y;

