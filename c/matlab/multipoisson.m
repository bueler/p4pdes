function multipoisson(N,Vnum,d)
% MULTIPOISSON  Solve poisson equation
%   - u_xx - u_yy = f
% on unit square (0,1) x (0,1) by multigrid on a finest grid which has
% N+1 x N+1 points where N has 2^d as a factor.  The grid at depth d
% coarsenings is solved by LU.  Default d yields coarsest grid having one
% unknown.
%
% Uses homogeneous Dirichlet b.c. and manufactured solution.
% Discrete equations are
%   - u_i-1,j - u_i+1,j - u_i,j-1 - u_i,j+1 + 4 u_i,j = h^2 f_i,j
% at interior points.  Smoother is red-black Gauss-Seidel.  Solves exactly
% (i.e. by LU decomposition using backslash operator) on coarsest grid.
%
% Goal is to reproduce table page 65 of Briggs et al (2000), "A Multigrid
% Tutorial," 2nd ed., SIAM Press.

  % get and check on depth d
  if nargin < 3
    d = log2(N) - 1;
    if mod(N,2^d) ~= 0,  error('N must be a power of two (N=2^d)'),  end
  else
    if mod(N,2^d) ~= 0,  error('N must have 2^d as a factor'),  end
  end

  % defaults to 15 V cycles
  if nargin < 2
    Vnum = 15;
  end

  % configure grid
  h = 1.0 / N;
  xaxis = 0:h:1;
  yaxis = xaxis;
  [x,y] = meshgrid(xaxis,yaxis);

  % do Vnum V-cycles
  u = zeros(size(x));
  ff = f(x,y);
  r = residual(N,h,u,ff);
  uex = uexact(x,y);
  fprintf('%3d   %.2e   %.2e\n',0,discrete2norm(r,h),discrete2norm(u - uex,h));
  for K = 1:Vnum
    %unew = vcycle(0,N,h,1,1,u,ff); % FIXME: solves exactly on finest grid
    unew = vcycle(1,N,h,1,1,u,ff); % FIXME
    r = residual(N,h,unew,ff);
    fprintf('%3d   %.2e   %.2e\n',K,discrete2norm(r,h),discrete2norm(unew - uex,h));
    u = unew;
  end

  % optionally visualize solution and error
  dograph = false;
  if dograph
    figure(1),
    zmin = min(min(uex));
    subplot(1,3,1), mesh(x,y,u), axis([0 1 0 1 zmin 0]), title('numerical u')
    subplot(1,3,2:3), imagesc(xaxis,yaxis,u-uex), title('u - uexact'), colorbar
  end
end


function unew = vcycle(level,N,h,nu1,nu2,u,f)

  if level > 0
    % relax nu1 times on fine grid
    for K = 1:nu1
      u = relax(N,h,u,f);
    end

    % compute fine-grid residual r
    r = residual(N,h,u,f);

    % restrict r to coarse grid
    rcoarse = fullweighting(N,h,r);

    % solve  A e = r  on coarser grid
    eguess = zeros(N/2+1,N/2+1);
    ecoarse = vcycle(level-1,N/2,2*h,nu1,nu2,eguess,rcoarse);

    % interpolated error e to fine grid, and update u
    % FIXME

    % relax nu2 times on fine grid
    for K = 1:nu2
      u = relax(N,h,u,f);
    end
    unew = u;
  else
    % solve exactly on coarsest grid
    unew = lusolve(N,h,f);
  end
end


function UU = relax(N,h,u,f)
% RELAX is one step of red-black Gauss-Seidel
  UU = zeros(size(u));
  hsqr = h * h;
  % red sweep
  for i = 2:2:N
    for j = 2:2:N
      UU(i,j) = (1/4) * ( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + hsqr * f(i,j) );
    end
  end
  for i = 3:2:N-1
    for j = 3:2:N-1
      UU(i,j) = (1/4) * ( u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) + hsqr * f(i,j) );
    end
  end
  % black sweep
  for i = 3:2:N-1
    for j = 2:2:N
      UU(i,j) = (1/4) * ( UU(i-1,j) + UU(i+1,j) + UU(i,j-1) + UU(i,j+1) + hsqr * f(i,j) );
    end
  end
  for i = 2:2:N
    for j = 3:2:N-1
      UU(i,j) = (1/4) * ( UU(i-1,j) + UU(i+1,j) + UU(i,j-1) + UU(i,j+1) + hsqr * f(i,j) );
    end
  end
end


function U = lusolve(N,h,f)
% LUSOLVE  Exactly solve discrete problem by storing in sparse matrices
% and using backslash.
  K = (N-1)*(N-1);
  A = spdiags(repmat(4,K,1),0,K,K);
  b = zeros(N-1,1);
  hsqr = h * h;
  for i = 1:N-1
    for j = 1:N-1
      k      = (N-1) * (i-1) + j;
      kleft  = (N-1) * (i-1) + j-1;
      kright = (N-1) * (i-1) + j+1;
      kup    = (N-1) * (i)   + j;
      kdown  = (N-1) * (i-2) + j;
      if j>1,    A(k,kleft)  = -1.0;  end
      if j<N-1,  A(k,kright) = -1.0;  end
      if i>1,    A(k,kdown)  = -1.0;  end
      if i<N-1,  A(k,kup)    = -1.0;  end
      b(k,1) = hsqr * f(i+1,j+1);
    end
  end
  uu = A \ b;
  %figure(99), spy(A)
  U = zeros(N+1,N+1);
  for i = 1:N-1
    for j = 1:N-1
      k = (N-1) * (i-1) + j;
      U(i+1,j+1) = uu(k);
    end
  end
end


function r = residual(N,h,u,f)
% RESIDUAL compute  r = f - A u
  r = zeros(size(u));
  hsqr = h * h;
  for i = 2:N
    for j = 2:N
      r(i,j) = f(i,j) + (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) - 4 * u(i,j)) / hsqr;
    end
  end
end


function wcoarse = fullweighting(N,h,w)
% FULLWEIGHTING  compute restriction operator (assuming zero along bdry)
% See formula bottom page 36 of Briggs et al (2000).
  if mod(N,2) ~= 0,  error('N must be even'),  end
  if any(size(w) ~= [N+1,N+1]),  error('w must be N+1 by N+1 array'),  end
  wcoarse = zeros(N/2+1,N/2+1);
  for i = 2:N/2
    for j = 2:N/2
      ii = 2*i-1;
      jj = 2*j-1;
      wc = (1/16) * ( w(ii-1,jj-1) + w(ii-1,jj+1) + w(ii+1,jj-1) + w(ii+1,jj+1) );
      wc = wc + (1/8) * ( w(ii,jj-1) + w(ii,jj+1) + w(ii-1,jj) + w(ii+1,jj) );
      wc = wc + (1/4) * w(ii,jj);
      wcoarse(i,j) = wc;
    end
  end
end


function z = discrete2norm(x,h)
  z = norm(x) * h;
end


function z = uexact(x,y)
  z = (x.^2 - x.^4) .* (y.^4 - y.^2);
end


function z = f(x,y)
  z = 2 * ( (1-6*x.^2) .* y.^2 .* (1-y.^2) + (1-6*y.^2) .* x.^2 .* (1-x.^2) );
end
