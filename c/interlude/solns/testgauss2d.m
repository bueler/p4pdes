function testgauss2d(n)
% TESTGAUSS2D  Run a test on degree n Gauss-Legendre quadrature on the reference
% Q^1 element in 2D, i.e. E_* = [-1,1] x [-1,1].  Test is whether a degree k
% polynomial in two variables is exactly integrated.  Uses tensor-product
% formulas.  Works for n=1,2,3,4.

if n~=floor(n), error('n must be an integer'), end
if (n < 1) || (n > 4), error('only n=1,2,3,4 are allowed'), end

z4a = sqrt((3/7) - (2/7)*sqrt(6/5));
z4b = sqrt((3/7) + (2/7)*sqrt(6/5));
w4a = (18+sqrt(30))/36;
w4b = (18-sqrt(30))/36;

z = {[0],
     [-1/sqrt(3),+1/sqrt(3)],
     [-sqrt(3/5),0,+sqrt(3/5)],
     [-z4b,-z4a,z4a,z4b]};

w = {[2],
     [1,1],
     [5/9,8/9,5/9],
     [w4b,w4a,w4a,w4b]};

fprintf('degree  difference  success\n')
for k=0:7
    % do quadrature
    tsum = 0.0;
    for r=1:n
        rsum = 0.0;
        for s=1:n
            rsum = rsum + w{n}(s) * integrand(k,z{n}(r),z{n}(s));
        end
        tsum = tsum + w{n}(r) * rsum;
    end
    % compare to exact integral and report
    err = abs(tsum - exact(k));
    if err <= (k+1)^2 * 10 * eps
        succstr = 'yes';
    else
        succstr = ' no';
    end
    fprintf('%d         %6.2e      %s\n',k,err,succstr)
end

function f = integrand(k,x,y)
    f = (1+x)^k + (1+y)^k;
end

function r = exact(k)
    r = 2^(k+3) / (k+1);
end

end

