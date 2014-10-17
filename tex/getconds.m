 % nonsymmetric matrix from first FD procedure
 M = eye(12);
 x = [-4, -9, 26, -9, -4];
 M(6, [2 5 6 7 10]) = x;
 M(7, [3 6 7 8 11]) = x;
 cond(M)
 
 % symmetric from second
 S = eye(12);
 A = 13/3; B = 3/2;
 S([6 7],[6 7]) = [A B; B A];
 cond(S)
