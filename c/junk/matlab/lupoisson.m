function lupoisson(N)
% LUPOISSON  Solve poisson equation
%   - u_xx - u_yy = f
% on unit square (0,1) x (0,1) by multigrid on problem using homogeneous
% Dirichlet b.c.  Problem has manufactured exact solution.
%
% Grid has N+1 x N+1 points.  Discrete system is solved exactly by LU.
% Compare MULTIPOISSON.
%
% Example:
%   >> lupoisson(16)
%   >> tic, multipoisson(256), toc
%   >> tic, lupoisson(256), toc

  % configure grid
  h = 1.0 / N;
  xaxis = 0:h:1;  yaxis = xaxis;
  [x,y] = meshgrid(xaxis,yaxis);

  % other initial fields and norms
  ff = f(x,y);
  uex = uexact(x,y);

  % solve
  u = lusolve(N,h,ff);

  % report norm of residual, and discretization error
  r = residual(N,h,u,ff);
  fprintf('      %.2e          %.2e\n',discrete2norm(r,h),discrete2norm(u-uex,h));

  % optionally visualize solution and error
  dograph = false;
  if dograph
    figure(1),
    zmin = min(min(uex));
    subplot(1,3,1), mesh(x,y,u), axis([0 1 0 1 zmin 0]), title('numerical u')
    subplot(1,3,2:3), imagesc(xaxis,yaxis,u-uex), title('u - uexact'), colorbar
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


function z = discrete2norm(x,h)
  z = norm(x) * h;
end


function z = uexact(x,y)
  z = (x.^2 - x.^4) .* (y.^4 - y.^2);
end


function z = f(x,y)
  z = 2 * ( (1-6*x.^2) .* y.^2 .* (1-y.^2) + (1-6*y.^2) .* x.^2 .* (1-x.^2) );
end
