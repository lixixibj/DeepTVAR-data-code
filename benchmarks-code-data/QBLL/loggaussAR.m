function LogL = loggaussAR(model, N, tildeN)
%function LogL = loggaussAR(model, N, tildeN)
%
% Computes the log probability density for the big Gaussian AR prior


% place the blocks one below the other
LogL = 0; 
for i=1:N
    tmpM = tridiagAR(model.T, model.phi_h(i), model.sigma2_h(i));    
    
    L = chol(tmpM);
    ok = L*(model.hs(i,:)' - model.h_0(i));
    LogL = LogL - 0.5*model.T*log(2*pi) + sum(log(diag(L))) - 0.5*(ok'*ok);
    
    %k2 = model.sigma2_h(i)/(1 - model.phi_h(i)^2);
    %ll2 = - 0.5*log(2*pi*k2) - ((model.hs(i,1) - model.h_0(i)).^2)/(2*k2);
    %for t=2:model.T
    %   k0 = model.h_0(i) + model.phi_h(i)*(model.hs(i,t-1) - model.h_0(i));
    %   ll2 = ll2 - 0.5*log(2*pi*model.sigma2_h(i)) - ((model.hs(i,t) - k0).^2)./(2*model.sigma2_h(i));
    %end
    %ll2 - ll
    %i
    %pause  
end

for j=1:tildeN
%    
    tmpM = tridiagAR(model.T, model.phi_delta(j), model.sigma2_delta(j));
    L = chol(tmpM);
    ok = L*(model.deltas(j,:)' - model.delta_0(j));
    LogL = LogL - 0.5*model.T*log(2*pi) + sum(log(diag(L))) - 0.5*(ok'*ok);
%    
end

