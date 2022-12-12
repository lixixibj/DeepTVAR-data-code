function logPrior = logmarginalizedNormalGam(phi, mu0, k0, alpha0, beta0)
%function logPrior = marginNormalGamma(phi, k0, alpha0, beta0)
%
%


N = length(phi); 

kN = k0 + N; 
alphaN = alpha0 + 0.5*N;

barphi = mean(phi); 
betaN = beta0 + 0.5*sum((phi - barphi).^2)  + (k0*N*((barphi-mu0)^2))/(2*kN); 


logPrior = gammaln(alphaN) - gammaln(alpha0)  + alpha0*log(beta0) - alphaN*log(betaN) ...
          + 0.5*log(k0/kN) - 0.5*N*log(2*pi); 
       