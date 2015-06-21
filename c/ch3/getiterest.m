function N = getiterest(rtol,lamBIG,lamSMALL)

kap = lamBIG/lamSMALL
gam = (sqrt(kap) - 1) / (sqrt(kap) + 1)

N = log(rtol/2) / log(gam);

for k=1:100
   if 2 / ( gam^k + gam^(-k) )  < rtol
      break;
   end
end
k
