function G = givensmat(omega, N, i, j) 
%function G = givensmat(omega, i, j) 


G = sparse(eye(N)); 

cc = cos(omega);
ss = sin(omega);

G(i,i) = cc;
G(j,j) = cc;

G(i,j) = ss;
G(j,i) = -ss;
