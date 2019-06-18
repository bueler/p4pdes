% BLOCKBIHARM   Extract blocks of biharmonic equation system matrix, and
% rebuild the block matrix
%       / Av |  0 \
%   H = |----|----|
%       \ -I | Av /
% Run biharm.c first with small LEV:
%   $ ./biharm -da_refine LEV -ksp_view_mat :foo.m:ascii_matlab
%   $ octave
%   >> foo
%   >> A = Mat...;
%   >> blockbiharm

Av = A(1:2:end,1:2:end);
Au = A(2:2:end,2:2:end);
Below = A(2:2:end,1:2:end);
Zero = A(1:2:end,2:2:end);

H = [Av Zero; Below Au];

norm(Zero)                                      % above diag should be zero
norm(diag(diag(Below)) - Below)                 % below diag should be diag
norm(Au - Av) / norm(Av)                        % diag blocks should be same
norm(real(sort(eig(A))) - real(sort(eig(H))))   % eigs should be same
norm(svd(A) - svd(H))                           % singular vals should be same

