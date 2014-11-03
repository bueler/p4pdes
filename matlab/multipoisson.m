function table = multipoisson(N,numcycles,d)
% MULTIPOISSON  Solve poisson equation
%   - u_xx - u_yy = f
% on unit square (0,1) x (0,1) by multigrid on problem using homogeneous
% Dirichlet b.c.  Problem has manufactured exact solution.
%
% The finest grid has N+1 x N+1 points where N has 2^d as a factor.  The
% grid at depth d coarsenings is solved exactly by LU.  Default d yields
% a coarsest grid having one unknown.
%
% Discrete equations are
%   - u_i-1,j - u_i+1,j - u_i,j-1 - u_i,j+1 + 4 u_i,j = h^2 f_i,j
% at interior points.  Smoother is red-black Gauss-Seidel.  Restriction
% operator is full weighting.  Interpolation operator is linear.
%
% Goal is to reproduce table page 65 of Briggs et al (2000), "A Multigrid
% Tutorial," 2nd ed., SIAM Press.
%
% Calling:
%   table = multipoisson(N,numcycles,d)
% where
%   N         : grid is N+1 by N+1
%   numcycles : do this many V-cycles  (default = 15)
%   d         : refinement depth; require 2^d | N  (default = log2(N)-1)
% and output is
%   table     : string to compare to table in Briggs et al (2000)
%
% Examples:
%   >> multipoisson(16)     # 17x17 grid and use 15 (default) V-cycles
%   >> multipoisson(8,50,1) # 9x9 grid; 50 V-cycles; coarse grid is 5x5

  % get and check on depth d
  if nargin < 3
    d = log2(N) - 1;
    if mod(N,2^d) ~= 0,  error('N must be a power of two (N=2^d)'),  end
  else
    if mod(N,2^d) ~= 0,  error('N must have 2^d as a factor'),  end
  end

  % defaults to 15 V cycles
  if nargin < 2
    numcycles = 15;
  end

  % configure grid
  h = 1.0 / N;
  xaxis = 0:h:1;  yaxis = xaxis;
  [x,y] = meshgrid(xaxis,yaxis);

  % initial u on fine grid
  % [which matches Briggs et al (2000) initial value?]
  %u = zeros(size(x));
  %u = rand(size(x));
  u = ( sin(16*pi*x) + sin(40*pi*x) ) .* ( sin(16*pi*y) + sin(40*pi*y) );

  % other initial fields and norms
  ff = f(x,y);
  r = residual(N,h,u,ff);
  uex = uexact(x,y);
  table = sprintf('\n');
  [s,rnorm,enorm] = report(0,r,u-uex,h);
  table = strcat(table,s);

  % do V-cycles
  for K = 1:numcycles
    fprintf('V-cycle %d:\n',K)
    u = vcycle(d,d,N,h,2,1,u,ff,true);
    r = residual(N,h,u,ff);
    [s,rnorm,enorm] = report(K,r,u-uex,h,rnorm,enorm);
    table = strcat(table,s);
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


function [s, rnorm, enorm] = report(K,r,udiff,h,rnormlast,enormlast)
  rnorm = discrete2norm(r,h);
  enorm = discrete2norm(udiff,h);
  if nargin > 4
    s = sprintf('%3d   %.2e   %.2f   %.2e   %.2f\n',...
                K,rnorm,rnorm/rnormlast,enorm,enorm/enormlast);
  else
    s = sprintf('%3d   %.2e          %.2e\n',K,rnorm,enorm);
  end
end


function s = indent(d,level)
  s = repmat('  ',1+d-level,1);
end


function unew = vcycle(d,level,N,h,nu1,nu2,u,f,verbose)
  exactoncoarsest = false;

  if ((level == 0) && (exactoncoarsest))
    % solve exactly on coarsest grid, so arguments nu1, nu2, and u
    %   are ignored
    if verbose
      fprintf('%ssolving exactly with LU on coarsest grid\n',indent(d,level))
    end
    unew = lusolve(N,h,f);
  else
    % relax nu1 times on fine grid
    for K = 1:nu1
      u = relax(N,h,u,f);
    end

    if level > 0
      if verbose
        fprintf('%sdescending to coarser level %d ...\n',indent(d,level),level-1)
      end

      % compute fine-grid residual r and restrict to coarse grid
      rfine = residual(N,h,u,f);
      rcoarse = fullweighting(N,rfine);

      % solve  A e = r  on coarser grid
      eguess = zeros(N/2+1,N/2+1);
      ecoarse = vcycle(d,level-1,N/2,2*h,nu1,nu2,eguess,rcoarse,verbose);

      % interpolated error e to fine grid, and update u
      efine = interpolate(N,ecoarse);
      u = u + efine;

      if verbose
        fprintf('%s... ascending\n',indent(d,level))
      end
    else
      if verbose
        fprintf('%ssolving with nu1+nu2 sweeps on coarsest grid\n',indent(d,level))
      end
    end

    % relax nu2 times on fine grid
    for K = 1:nu2
      u = relax(N,h,u,f);
    end
    unew = u;

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
% [does this match Briggs et al (2000) meaning of discrete 2-norm of residual?]
  r = zeros(size(u));
  hsqr = h * h;
  for i = 2:N
    for j = 2:N
      r(i,j) = f(i,j) + (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) - 4 * u(i,j)) / hsqr;
    end
  end
end


function wfine = interpolate(Nfine,w)
% INTERPOLATE  compute interpolation operator
% See formulas page 35 of Briggs et al (2000).
  if mod(Nfine,2) ~= 0
    error('Nfine must be even'),  end
  if any(size(w) ~= [Nfine/2+1,Nfine/2+1])
    error('w must be Nfine/2+1 by Nfine/2+1 array'),  end
  wfine = zeros(Nfine+1,Nfine+1);
  for i = 1:2:Nfine+1
    for j = 1:2:Nfine+1
      ii = (i-1)/2+1;
      jj = (j-1)/2+1;
      wfine(i,j) = w(ii,jj);
      if i>1
        wfine(i-1,j)   = (w(ii-1,jj) + w(ii,jj)) / 2;
      end
      if j>1
        wfine(i,j-1)   = (w(ii,jj-1) + w(ii,jj)) / 2;
      end
      if (i>1) && (j>1)
        wfine(i-1,j-1) = (w(ii-1,jj-1) + w(ii-1,jj) + w(ii,jj-1) + w(ii,jj)) / 4;
      end
    end
  end
end


function wcoarse = fullweighting(Nfine,w)
% FULLWEIGHTING  compute restriction operator (assuming zero along bdry)
% See formula bottom page 36 of Briggs et al (2000).
  if mod(Nfine,2) ~= 0
    error('Nfine must be even'),  end
  if any(size(w) ~= [Nfine+1,Nfine+1])
    error('w must be Nfine+1 by Nfine+1 array'),  end
  wcoarse = zeros(Nfine/2+1,Nfine/2+1);
  for i = 2:Nfine/2
    for j = 2:Nfine/2
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
