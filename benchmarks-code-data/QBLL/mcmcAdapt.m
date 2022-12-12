function [model PropDist samples accRates] = mcmcAdapt(model, ops, Langevin)
%
%


if strcmp(model.useFactorModel, 'no') == 1
    model.L = eye(model.K);
    model.Weights = eye(model.K);
    model.sigma2 = 0; 
    model.Ft = model.Resid; 
    for t=1:model.T
       indNans = find(isnan(model.Resid(:,t)));
       indObs = find(~isnan(model.Resid(:,t)));
       if length(indNans) < model.N
           model.Ft(indNans,t) = mean(model.Resid(indObs,t));
       else
           model.Ft(indNans,t) = 0; 
       end 
    end
else
    v = 0;
    for t=1:model.T
       indObs = find(~isnan(model.Resid(:,t)));
       if length(indObs)>1
          v = v + var(model.Resid(indObs,t));
       end
    end
    if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
       v = zeros(1,model.N);
       for nn=1:model.N
       indObs = find(~isnan(model.Resid(nn,:)));
       if length(indObs)>1
          v(nn) = var(model.Resid(nn,indObs));
       end
       end
       model.sigma2 = 0.5*v; 
       model.sigma2
    else
    model.sigma2 = 0.5*(v/model.T); 
    end
end
 

BurnInIters = ops.Burnin; 
Iters = ops.T; 

n = model.K*model.T + model.tildeK*model.T;

model.auxLikVar = 0.1;
if strcmp(model.sampleFactorsByGibbs, 'no') == 1
  model.deltaFactors = 0.01*ones(model.T, 1); 
end
  
epsilon = 0.15;
nextbreak = 0;
cnt = 0;

range = 0.05; 
if isfield(model,'opt')
   opt = model.opt;
   range = 0; 
else
   if Langevin == 1 
      opt = 0.54;
   else
      opt = 0.25;
   end
end

PropDist.phi = 0.1;
epsilon = 0.15;
nextbreak = 0;
cnt = 0;
%model.sampleSigma2 = 0; 
while 1
    %
    
    [model samples accRates] = mcmcTrainVar(model, PropDist, ops, Langevin);
    
    accRateF = accRates.F;
    accRateFt = accRates.Ft;
    accRatePhi = accRates.Phi;
    
    %fprintf(1,'------ ADAPTION STEP #%2d ------ \n',cnt+1); 
    if ops.disp == 1
        if strcmp(model.sampleFactorsByGibbs, 'no') == 1
          fprintf(1,'Adapt-step=%d, AccRate ft=%f, AccRate for hs&deltas=%f, AccRate for Phis=%f, Average log lik=%f\n',cnt, median(accRateFt), accRateF, accRatePhi, mean(samples.LogL));    
        else
          fprintf(1,'Adapt-step=%d, AccRate for hs&deltas=%f, AccRate for Phis=%f, Average log lik=%f\n',cnt, accRateF, accRatePhi, mean(samples.LogL));    
        end 
    end
    
    % do always op.minAdapIters iterations
    if cnt > ops.minAdapIters
      % if you got a descent acceptance rate, then stop
    if (accRateF > ((opt-0.05)*100)) & (accRateF < ((opt+0.15)*100)) &  (accRatePhi>15) & (median(accRateFt) > 40)
        if nextbreak == 2
           disp('END OF ADAPTION: acceptance rates OK');
           break; 
        else
           nextbreak = nextbreak + 1;
        end
    else
        nextbreak = 0;
    end
    end
    
    cnt = cnt + 1;
    % do not allow more than 500 iterations when you adapt the proposal distribution
    if cnt == ops.minAdapIters
        warning('END OF ADAPTION: acceptance rates were not all OK');
        break;
    end
  
    %%%%%%%%%%%% ADAPT AUXILIARY LIKELIHOOD VARIANCE %%%%%%%%%%%
    if (accRateF > (100*(opt+range))) | (accRateF < (100*(opt-range)))
    %    
       model.auxLikVar = model.auxLikVar + (epsilon*((accRateF/100 - opt)/opt))*model.auxLikVar;
    %   
    end
    
    if strcmp(model.useFactorModel, 'no') == 1
      for t=1:model.T 
        indNans = find(isnan(model.Resid(:,t)));
        if length(indNans)>0
        if (accRateFt(t) > (100*(opt+range))) | (accRateFt(t) < (100*(opt-range)))
        %    
           model.deltaFactors(t) = model.deltaFactors(t) + (epsilon*((accRateFt(t)/100 - opt)/opt))*model.deltaFactors(t);
        %   
        end
        end
       end
    elseif (strcmp(model.sampleFactorsByGibbs, 'no') == 1)
        for t=1:model.T
        indObs = find(~isnan(model.Resid(:,t)));
        if length(indObs) > 0
        if (accRateFt(t) > (100*(opt+range))) | (accRateFt(t) < (100*(opt-range)))
        %    
           model.deltaFactors(t) = model.deltaFactors(t) + (epsilon*((accRateFt(t)/100 - opt)/opt))*model.deltaFactors(t);
        %   
        end
        end
        end
       %
    end
   
    if accRatePhi > 35
      % incease the covariance to reduce the acceptance rate
      PropDist.phi = PropDist.phi + epsilon*PropDist.phi;
    end
    if accRatePhi < 20
       % decrease the covariance to incease the acceptance rate
       PropDist.phi = PropDist.phi - epsilon*PropDist.phi;    
    %
    end  
    %
end
