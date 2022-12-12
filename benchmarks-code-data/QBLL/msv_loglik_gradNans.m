function [loglik, gradYtnans] = msv_loglik_gradNans(yt, indNans, omegas, hs, Givset)
%
%

N = length(yt); 
% it computes y*U 
v = yt';
for k=1:size(Givset,1)
    c = cos(omegas(k));
    s = sin(omegas(k));    
    tmp = v;
    v(Givset(k,1)) = c*tmp(Givset(k,1)) - s*tmp(Givset(k,2));
    v(Givset(k,2)) = s*tmp(Givset(k,1)) + c*tmp(Givset(k,2));
end
% computes y^T*U*Lambda^{-1} 
v = v.*exp(-0.5*hs');
loglik = - 0.5*N*log(2*pi) - 0.5*sum(hs) - 0.5*(v*v');
v = v.*exp(-0.5*hs');

% y^T*U*Lambda^{-1}*U^T= y^T*Sigma^{-1}
for k=size(Givset,1):-1:1
    c = cos(omegas(k));
    s = sin(omegas(k));    
    tmp = v;
    v(Givset(k,1)) = c*tmp(Givset(k,1)) + s*tmp(Givset(k,2));
    v(Givset(k,2)) = - s*tmp(Givset(k,1)) + c*tmp(Givset(k,2));
end
% computes the final derivaives  
gradYtnans = - v(indNans)';