function [Ne, Nr] = getiterest(rtol,lamBIG,lamSMALL)

kap = lamBIG/lamSMALL
gam = (sqrt(kap) - 1) / (sqrt(kap) + 1)

Ne = -1;
for k=1:100
   if (Ne < 0) & (2 * gam^k  < rtol)
      Ne = k;
   end
   if 2 * sqrt(kap) * gam^k  < rtol  % factor of sqrt(kap) from converting |e|_A to |r|_2
      break;
   end
end
Nr = k;
