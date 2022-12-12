function [model samples accRates] = mcmcTrainVarF(model, PropDist, trainOps, Langevin)
% Inputs: 
%         -- model: the structure that contains the likelihood and GP
%                    parameters as well as the priors for all these
%                    quantities
%         -- PropDist: a stucture that defines the functional form of the proposal distribution
%         -- trainOps: user defined options about the burn-in and sampling iterations
%                      and others (see demos)
%
% Outputs: model: 
%         -- model: as above. The outputed model is updated to contain the
%                   parameters values of the final MCMC iteration
%                   parameters as well as the priors
%         -- PropDist: as above. PropDist can be updated (compared to the input one) 
%                     due to the update of the kernel parameters that
%                     influence the proposal 
%         -- samples: the structure that contrains the samples 
%         -- accRates: acceptance rates 
%


BurnInIters = trainOps.Burnin; 
Iters = trainOps.T; 
StoreEvery = trainOps.StoreEvery;
num_stored = floor(Iters/StoreEvery);

n = model.K*model.T + model.tildeK*model.T;

samples.F = zeros(num_stored, n);
samples.Ft = zeros(model.K, model.T, num_stored);
samples.Phi_h = zeros(num_stored, model.K);
samples.Phi_delta = zeros(num_stored, model.tildeK);
samples.h_0 = zeros(num_stored, model.K);
samples.delta_0 = zeros(num_stored, model.tildeK);
samples.sigma2_h = zeros(num_stored, model.K);
samples.sigma2_delta = zeros(num_stored, model.tildeK);
if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
samples.sigma2 = zeros(num_stored, model.N); 
else
samples.sigma2 = zeros(1, num_stored);
end
samples.Weights = zeros(model.N, model.K, num_stored);
samples.LogL = zeros(1, num_stored);
estimSigma=zeros(model.N, model.N, model.T);

cnt = 0;
acceptF = 0;
acceptFt = zeros(model.T, 1);
acceptPhi = 0;

% Create the big contaminated inverse 
% covariance matrix of all hs and deltas
% tridiagonal matrix
%W = spalloc(n,n,3*n-2);
iAll = []; 
jAll = [];
en = 0; 
for k=1:model.K
    iAll = [iAll, (1+en):(model.T+en)]; 
    jAll = [jAll, (1+en):(model.T+en)];
    iAll = [iAll, (1+en):(model.T-1+en)]; 
    jAll = [jAll, (2+en):(model.T+en)];
    iAll = [iAll, (2+en):(model.T+en)]; 
    jAll = [jAll, (1+en):(model.T-1+en)];
    en = en + model.T;  
end
for k=1:model.tildeK  
    iAll = [iAll, (1+en):(model.T+en)]; 
    jAll = [jAll, (1+en):(model.T+en)];
    iAll = [iAll, (1+en):(model.T-1+en)]; 
    jAll = [jAll, (2+en):(model.T+en)];
    iAll = [iAll, (2+en):(model.T+en)]; 
    jAll = [jAll, (1+en):(model.T-1+en)];
    en = en + model.T;  
end
%W = sparse(iAll, jAll, ones(3*n - 2*(model.K + model.tildeK), 1)); 
%Wtmp = sparse(iAll, jAll, ones(3*n - 2*(model.K + model.tildeK), 1)); 

II = []; 
JJ = [];
II = [II, 1:model.T]; 
JJ = [JJ, 1:model.T];
II = [II, 1:model.T-1]; 
JJ = [JJ, 2:model.T];
II = [II, 2:model.T]; 
JJ = [JJ, 1:model.T-1];

model.phi_h = (exp(model.tildephi_h) - 1)./(exp(model.tildephi_h) + 1);
model.phi_delta = (exp(model.tildephi_delta) - 1)./(exp(model.tildephi_delta) + 1);

% place the blocks one below the other
nonZeros = [];
for i=1:model.K
    %tmpM = tridiagAR(model.T, model.phi_h(i), model.sigma2_h(i));   
    %st = (i-1)*model.T + 1;
    %en = i*model.T;
    %W(st:en,st:en) = tmpM;
    
    d = ((1+model.phi_h(i)^2)/model.sigma2_h(i))*ones(1,model.T);
    d(1) = 1/model.sigma2_h(i);
    d(end) = 1/model.sigma2_h(i);
    nonZeros = [nonZeros, d];
    offd = -model.phi_h(i)/model.sigma2_h(i);
    nonZeros = [nonZeros, offd*ones(1, 2*model.T - 2)];  
end


for j=1:model.tildeK
    %st = model.K*model.T + (j-1)*model.T + 1;
    %en = model.K*model.T + j*model.T;
    %tmpM = tridiagAR(model.T, model.phi_delta(j), model.sigma2_delta(j)); 
    %W(st:en,st:en) = tmpM;
     
    d = ((1+model.phi_delta(j)^2)/model.sigma2_delta(j))*ones(1,model.T);
    d(1) = 1/model.sigma2_delta(j);
    d(end) = 1/model.sigma2_delta(j);
    nonZeros = [nonZeros, d];
    offd = -model.phi_delta(j)/model.sigma2_delta(j);
    nonZeros = [nonZeros, offd*ones(1, 2*model.T - 2)];  
end
W = sparse(iAll, jAll, nonZeros); 

Lfree = jitterChol(W); 

% add the inverse of the auxiliary data variance in the diagonal
Wonly = W;
W = W + diag(sparse(1./model.auxLikVar*ones(1,n)));
L = jitterChol(W)';

% concatinate the hs and deltas in a single vector to be sampled
tmp = model.hs';
F = tmp(:); 
tmp = model.deltas';
F = [F; tmp(:)];

% concantinate the mean values
tmp = repmat(model.h_0,model.T,1);
mu = tmp(:);
tmp = repmat(model.delta_0,model.T,1);
mu = [mu; tmp(:)];

% log likelihood for phis
%oldLogLphi = loggaussAR(model, model.K, model.tildeK); 

ok = Lfree*(F - mu);
oldLogLphi = - 0.5*(model.T*(model.K+model.tildeK))*log(2*pi) + sum(log(diag(Lfree))) - 0.5*(ok'*ok);

% log prior for phis
% log prior for phis
if strcmp(model.exchangeablePriorphi, 'yes') 
   oldPriorphi_h = logmarginalizedNormalGam(model.tildephi_h, model.priorPhi_h.mu0, model.priorPhi_h.k0, model.priorPhi_h.alpha0, model.priorPhi_h.beta0); 
   oldPriorphi_delta = logmarginalizedNormalGam(model.tildephi_delta, model.priorPhi_delta.mu0, model.priorPhi_delta.k0, model.priorPhi_delta.alpha0, model.priorPhi_delta.beta0);
else
   oldPriorphi_h = logNormal(model.tildephi_h, model.priorPhi_h.mu0, model.priorPhi_h.s2); 
   oldPriorphi_delta = logNormal(model.tildephi_delta, model.priorPhi_delta.mu0, model.priorPhi_delta.s2); 
end 
oldPriorphi = oldPriorphi_h + oldPriorphi_delta;

SS = eye(model.K);
 

if strcmp(model.diagonalFinalLikelihoodNoise, 'yes') 
   num_OfNonNans = zeros(1,model.N);
   for i=1:model.N  
    num_OfNonNans(i) = sum(~isnan(model.Resid(i,:)));  
   end
else
   num_OfNonNans = sum(~isnan(model.Resid(:)));    
end

model.num_OfNonNans = num_OfNonNans;


if Langevin == 1 
    gradDeltas = zeros( size(model.deltas) ); 
    gradHs = zeros( size(model.hs) );
    
    derF = zeros(length(F),1); 
    derFnew = zeros(length(F),1); 
end

for it = 1:(BurnInIters + Iters) 
%


%   KATERINA ADDED: COMPUTE TV SIGMA draw as it's needed to draw VAR coefficients    
       LW = model.L.*model.Weights;

        for t=1:model.T
            G = eye(model.K);
            for k=1:size(model.Givset,1)
                Gtmp = givensmat(model.omegas(k, t), model.K, model.Givset(k,1), model.Givset(k,2));
                G = Gtmp*G;
            end
            estimSigma(:,:,t) = LW*(G*diag(exp(model.hs(:,t)))*G')*LW'+diag(mean(model.sigma2,1));
        end  

   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                             Step 0                                            %
%%%                                 KATERINA sample VAR coefficients                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

         if model.VarConPara == 1 % constant parameters
         [Y,XX]=transform_tvsigma(model.VarX,model.VarY,estimSigma);

         BB = draw_fbeta( model.VarPriorVar,model.VarPriorMean, XX,Y,model.VarLag,1,model.VarCheckStability,model.StabilityThreshold);
         residuals=transform_fbeta(model.VarX,model.VarY,BB);
         
         model.Resid=residuals';
         %plot(residuals)

         elseif model.VarConPara == 0 && model.VarTvAll == 1 % ALL TV parameters

                 if strcmp(model.VarTvEstimation,'kernel')
                 [Y,XX]=transform_tvsigma(model.VarX,model.VarY,estimSigma);
                 BB = draw_TVbeta( model.VarPriorVar,model.VarPriorMean,model.VarKernelW,XX,Y,model.VarLag,model.VarCheckStability,model.StabilityThreshold);
                 residuals=transform_tvbeta(model.VarX,model.VarY,BB);
                 model.Resid=residuals';

                 elseif strcmp(model.VarTvEstimation,'random_walk')
                     
                 %draw the betas
                 if model.VarDiagStates==1
                 model.VarQ0=diag(diag(model.VarQ0));
                 end
                 [BB,residuals,~,~]=carterkohn_coef(model.VarY,model.VarX,model.VarQ0,estimSigma,(model.VarBB0(:))',model.VarP00,model.VarLag,model.VarCheckStability,model.StabilityThreshold,10);
                 % residuals=transform_tvbeta(model.VarX,model.VarY,BB');
                 % can also compute this way but it's probably cheaper
                 % within the KF recursions
                 model.Resid=residuals';

                 %draw the Qs
                 errorQ=diff(BB);
                 scaleQ=(errorQ'*errorQ)+model.VarQ0;
                 if model.VarDiagStates==0
                 model.VarQ0=iwpq(model.T-1+model.VarT0, invpd(scaleQ));
                 else
                 model.VarQ0=iwpq(model.T-1+model.VarT0, invpd(diag(diag(scaleQ))));    
                 end
                 end
                 
             elseif model.VarConPara == 0 && model.VarTvAll == 0 % ONLY intecepts vary
             VarXnoC=model.VarX(:,2:end);
             Ytilda=transform_fbeta(VarXnoC,model.VarY,model.Bconstant);
             [Y,XX]=transform_tvsigma(ones(model.T, 1),Ytilda,estimSigma);

                if strcmp(model.VarTvEstimation,'kernel')
             %draw the TV intercepts
             TvIntercept = draw_TVbeta( model.VarPriorVar(1,1),model.VarPriorMean(1,:),model.VarKernelW,XX,Y,model.VarLag,0,0);
             %TvIntercept = draw_TVintercept( model.VarPriorVar(1,1),model.VarPriorMean(1,:),model.VarKernelW,Ytilda');
             
                elseif strcmp(model.VarTvEstimation,'random_walk')
             %draw the TV intercepts with KF
             locationC = locateC( model.N, model.VarLag );
                    if model.VarDiagStates==1
                 model.VarQ0=diag(diag(model.VarQ0));
                    end
             [TvIntercept,~,~,~]=carterkohn_coef(Ytilda,ones(model.T,1),model.VarQ0(locationC,locationC),estimSigma,model.VarB0(locationC),model.VarP00(locationC,locationC),model.VarLag,0,0,10);
             %draw the Qs
                 errorQ=diff(TvIntercept);
                 scaleQ=(errorQ'*errorQ)+model.VarQ0(locationC,locationC);
                    if model.VarDiagStates==0
                 model.VarQ0(locationC,locationC)=iwpq(model.T-1+model.VarT0, invpd(scaleQ));
                    else
                 model.VarQ0(locationC,locationC)=iwpq(model.T-1+model.VarT0, invpd(diag(diag(scaleQ))));    
                    end
                 TvIntercept=TvIntercept';
             end
             %draw the fixed Bs
             VarYnoC = model.VarY-TvIntercept';
             [Y,XX]=transform_tvsigma(model.VarX(:,2:end),VarYnoC,estimSigma);
             model.Bconstant = draw_fbeta( model.VarPriorVar(2:end,2:end),model.VarPriorMean(2:end,:), XX,Y,model.VarLag,0,model.VarCheckStability,model.StabilityThreshold);
             BB=model.Bconstant;
             residuals=(model.VarY-TvIntercept'-model.VarX(:,2:end)*model.Bconstant);
             model.Resid=residuals';
         end
  
   
   if strcmp(model.useFactorModel, 'yes') 
   %     
      if strcmp(model.sampleFactorsByGibbs, 'yes') == 1
      %    
        % STEP 1: Sample latent factor variables f_t given the Sigma_t and the partial observed data 
        for t=1:model.T 
            ind = find(~isnan(model.Resid(:,t)));
            if length(ind) > 0
        
               % compute the Sigma_t^(-1) matrix (based on the givens representation)
               invSigmaHalf = zeros(model.K,model.K);
               for k=1:model.K 
                  [tmp, V] = msv_loglik(SS(:,k), model.omegas(:,t), model.hs(:,t), model.Givset);
                  invSigmaHalf(k,:) = V;
               end
               invSigma = invSigmaHalf*invSigmaHalf';     
              
               %if it > 100
               %  G = eye(model.K);
               %  for k=1:size(model.Givset,1)
               %     Gtmp = givensmat(model.omegas(k, t), model.K, model.Givset(k,1), model.Givset(k,2));
               %     G = Gtmp*G;
               %  end
               %  tmp = G*diag(exp(-model.hs(:,t)))*G';
               %  invSigma
               %  tmp
               %  invSigma - tmp
               %  if it > 110
               %     return;
               %  end
               %end
        
               Lobs = model.L(ind,:).*model.Weights(ind,:); 
               M = (Lobs'*Lobs) + model.sigma2*invSigma; 
               cholM = jitterChol(M);
               model.Ft(:,t) = sqrt(model.sigma2)*(cholM\randn(model.K,1)) + ((cholM\(cholM'\(Lobs'*model.Resid(ind,t)))));
            else
               v = randn(model.K,1).*exp(0.5*model.hs(:,t));
               for k=size(model.Givset,1):-1:1
                   c = cos(model.omegas(k,t));
                   s = sin(model.omegas(k,t));    
                   tmp = v;
                   v(model.Givset(k,1)) = c*tmp(model.Givset(k,1)) + s*tmp(model.Givset(k,2));
                   v(model.Givset(k,2)) = - s*tmp(model.Givset(k,1)) + c*tmp(model.Givset(k,2));
               end
               model.Ft(:,t) = v;      
            end
        end
      %  
      else
          
        % 
        for t=1:model.T 
            ind = find(~isnan(model.Resid(:,t)));
            if length(ind) > 0
               lognewlambdas = - model.hs(:,t) - log(model.deltaFactors(t)) + log( 2*exp(model.hs(:,t)) + model.deltaFactors(t) ); 
               %lognewlambdas = - model.hs(:,t)  + log( exp(log(2) + model.hs(:,t) - logdelta) + 1 ); 
          
               Lobs = model.L(ind,:).*model.Weights(ind,:); 
               
               rt = model.Resid(ind,t) - Lobs*model.Ft(:,t); 
               if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
                 gradFt = (Lobs'*((1./model.sigma2(ind))'.*rt));   
                 %LLobs = bsxfun(@times,Lobs, (1./sqrt(model.sigma2(ind)))');
                 %rrt = model.Resid(ind,t)./(sqrt(model.sigma2(ind))') - LLobs*model.Ft(:,t);
                 %gradFt = LLobs'*rrt;           
               else
                 gradFt = (1/model.sigma2)*(Lobs'*rt); 
               end 
               Zt = model.Ft(:,t) + (model.deltaFactors(t)/2)*gradFt + sqrt(model.deltaFactors(t)/2)*randn(model.K,1);
               
               v = Zt;
               for k=1:size(model.Givset,1)
                   c = cos(model.omegas(k,t));
                   s = sin(model.omegas(k,t));    
                   tmp = v;
                   v(model.Givset(k,1)) = c*tmp(model.Givset(k,1)) - s*tmp(model.Givset(k,2));
                   v(model.Givset(k,2)) = s*tmp(model.Givset(k,1)) + c*tmp(model.Givset(k,2));
               end
               v = v.*exp(-0.5*lognewlambdas);
               keep = randn(model.K,1); 
               v = ((2/model.deltaFactors(t))*v  +  keep).*exp(-0.5*lognewlambdas);
               for k=size(model.Givset,1):-1:1
                   c = cos(model.omegas(k,t));
                   s = sin(model.omegas(k,t));    
                   tmp = v;
                   v(model.Givset(k,1)) = c*tmp(model.Givset(k,1)) + s*tmp(model.Givset(k,2));
                   v(model.Givset(k,2)) = - s*tmp(model.Givset(k,1)) + c*tmp(model.Givset(k,2));
               end
               Ftnew = v;
          
               if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
                 rtnew = model.Resid(ind,t)  -  Lobs*Ftnew;
                 gradFtnew = (Lobs'*((1./model.sigma2(ind))'.*rtnew));
     
                 oldlikft = -0.5*(rt'*((1./model.sigma2(ind))'.*rt)); 
                 newlikft = -0.5*(rtnew'*((1./model.sigma2(ind))'.*rtnew));        
               else
                 rtnew = model.Resid(ind,t)  -  Lobs*Ftnew;
                 gradFtnew = (1/model.sigma2)*(Lobs'*rtnew);
     
                 oldlikft = -(0.5/model.sigma2)*(rt'*rt); 
                 newlikft = -(0.5/model.sigma2)*(rtnew'*rtnew);  
               end
               
               
               corrFactor = - (Zt' - model.Ft(:,t)')*gradFt + (Zt' - Ftnew')*gradFtnew;
               corrFactor = corrFactor - (model.deltaFactors(t)/4)*(gradFtnew'*gradFtnew - gradFt'*gradFt);
   
               [accept, uprob] = metropolisHastings(newlikft  + corrFactor, oldlikft, 0, 0);  
      
               if (it > BurnInIters) 
                   acceptFt(t) = acceptFt(t) + accept;
               end 
               if accept == 1
                   model.Ft(:,t) = Ftnew;
               end       
               %           % second way to get the sample (for debugging)
               %           if it > 9
               %              G = eye(model.K);
               %              for k=1:size(model.Givset,1)
               %                 Gtmp = givensmat(model.omegas(k, t), model.K, model.Givset(k,1), model.Givset(k,2));
               %                 G = G*Gtmp;
               %              end
               %              tmp = G*diag(exp( model.hs(:,t) + log(model.deltaFactors(t)) - log( 2*exp(model.hs(:,t)) + model.deltaFactors(t) )  ))*G';
               %              yy = tmp*(2/model.deltaFactors(t))*Zt  +  G*diag(exp( 0.5*(model.hs(:,t) + log(model.deltaFactors(t)) - log( 2*exp(model.hs(:,t)) + model.deltaFactors(t) ))  ))*keep;
               %              model.Ft
               %              lognewlambdas
               %              yy
               %              Ftnew 
               %              yy - Ftnew
               %           end  
            else 
               v = randn(model.K,1).*exp(0.5*model.hs(:,t));
               for k=size(model.Givset,1):-1:1
                   c = cos(model.omegas(k,t));
                   s = sin(model.omegas(k,t));    
                   tmp = v;
                   v(model.Givset(k,1)) = c*tmp(model.Givset(k,1)) + s*tmp(model.Givset(k,2));
                   v(model.Givset(k,2)) = - s*tmp(model.Givset(k,1)) + c*tmp(model.Givset(k,2));
               end
               model.Ft(:,t) = v;      
            end
        end
      end    
      
      % Sample the weigths in the factor loadings matrix
      for nnn=1:model.N
      %    
          ind = find(~isnan(model.Resid(nnn,:)));
          Yn = model.Resid(nnn, ind); 
          Phi =  diag(model.L(nnn,:))*model.Ft(:,ind);
          PPhi = Phi*Phi'; 
          if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
             Smatrix = PPhi + (model.sigma2(nnn)/model.sigma2weights)*eye(model.K);
             Lmatrix = jitterChol(Smatrix);
             %invLmatrix = Lmatrix\eye(model.K);
          
             tmp = sqrt(model.sigma2(nnn))*(Lmatrix\randn(model.K,1)) + (Lmatrix\(Lmatrix'\(Phi*Yn(:))));  
             model.Weights(nnn,:) = tmp'; 
          else
             Smatrix = PPhi + (model.sigma2/model.sigma2weights)*eye(model.K);
          
             Lmatrix = jitterChol(Smatrix);
             %invLmatrix = Lmatrix\eye(model.K);
          
             tmp = sqrt(model.sigma2)*(Lmatrix\randn(model.K,1)) + (Lmatrix\(Lmatrix'\(Phi*Yn(:))));  
             model.Weights(nnn,:) = tmp'; 
          end
      %    
      end
      
      %model.Weights = eye(model.N);
      %model.Weights
      
      % Sample also the noise variance sigma2 (Gibss step)
      if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
        alpha_post = model.priorSigma2.alpha0 + num_OfNonNans/2;
        beta_post = model.priorSigma2.beta0 + 0.5*nansum( (model.Resid - (model.L.*model.Weights)*model.Ft).^2, 2)'; 
        model.sigma2 = 1./gamrnd(alpha_post, 1./beta_post);  
      else
        alpha_post = model.priorSigma2.alpha0 + num_OfNonNans/2;
        beta_post = model.priorSigma2.beta0 + 0.5*nansum(nansum( (model.Resid - (model.L.*model.Weights)*model.Ft).^2 ))'; 
        model.sigma2 = 1/gamrnd(alpha_post, 1/beta_post);
      end
      % some bound for numerical stability
      %model.sigma2(model.sigma2<1e-10) = 1e-10;
   %    
   else % just simple MSV model where we sample the missing values
   %   

   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                THE REST OF THE CODE DEALS WITH MSV                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       for t=1:model.T
       %    
          indNans = find(isnan(model.Resid(:,t)));
          indObs = find(~isnan(model.Resid(:,t)));
          if length(indNans) > 0
          %    
             [oldlikyt, gradYtnansold] = msv_loglik_gradNans(model.Ft(:,t), indNans, model.omegas(:,t), model.hs(:,t), model.Givset);  
            
             ynanmu = model.Ft(indNans,t) + (model.deltaFactors(t)/2)*gradYtnansold;
            
             Ytnansnew = ynanmu + sqrt(model.deltaFactors(t))*randn(length(indNans),1);
            
             Ftnew = model.Ft(:,t); 
             Ftnew(indNans) = Ytnansnew; 
            
             [newlikyt, gradYtnansnew] = msv_loglik_gradNans(Ftnew, indNans, model.omegas(:,t), model.hs(:,t), model.Givset); 
             
             ynanmunew = Ytnansnew + (model.deltaFactors(t)/2)*gradYtnansnew;
             
             %% compute the langevin proposal terms in the M-H ratio
             tmp = Ytnansnew - ynanmu; 
             proposal_new_given_old = - (0.5/model.deltaFactors(t))*(tmp'*tmp);
             tmp = model.Ft(indNans,t) - ynanmunew;
             proposal_old_given_new = - (0.5/model.deltaFactors(t))*(tmp'*tmp);
        
             corrFactor = proposal_old_given_new - proposal_new_given_old;
                  
             [accept, uprob] = metropolisHastings(newlikyt + corrFactor, oldlikyt, 0, 0);  
      
             if (it > BurnInIters) 
                acceptFt(t) = acceptFt(t) + accept;
             end 
             if accept == 1
                model.Ft(indNans,t) = Ytnansnew;
             end
          else
          % 
             % somehow you *always accept* if there are not missing values
             if (it > BurnInIters) 
                acceptFt(t) = acceptFt(t) + 1;
             end 
          %   
          end
       end
   end
 
      
    % perform an evaluation of the likelihood p(Y | F) 
    oldloglik = 0;
    if Langevin == 1
        for t=1:model.T
          [tmp1, tmp2, tmpgradOmega, tmpgradHs] = msv_loglik(model.Ft(:,t), model.omegas(:,t), model.hs(:,t), model.Givset);
          gradDeltas(:,t) = tmpgradOmega'; 
          gradDeltas(:,t) = gradDeltas(:,t).*(pi*( exp(model.deltas(:,t))./( (1 + exp(model.deltas(:,t))).^2 ) ) ); 
          gradHs(:,t) = tmpgradHs';
          oldloglik = oldloglik + tmp1; 
        end
    else    
       for t=1:model.T
         oldloglik = oldloglik + msv_loglik(model.Ft(:,t), model.omegas(:,t), model.hs(:,t), model.Givset);
       end
    end
    
    storeloglik =  oldloglik;
     
    % STEP 2: Samples auxiliry variables
    % sample auxiliary data points
    if Langevin == 1
        % old gradient 
        tmp = gradHs';  
        derF(1:model.K*model.T) = tmp(:); 
        tmp = gradDeltas';
        derF((model.K*model.T+1):end) = tmp(:);   
        
        Z = F + (model.auxLikVar.*(derF)) + sqrt(model.auxLikVar).*randn(n,1);
    else
        Z = F + sqrt(model.auxLikVar)*randn(n,1);
    end
     
    % STEP 3: Sample hs and deltas all together 
  
    % Propose new values for hs and deltas given the proposed munew  
    
    %Fnew = mu + (1./model.auxLikVar)*(L'\(L\(Z - mu))) + L'\randn(n,1);
    Fnew = L'\(  L\( (1./model.auxLikVar)*Z + Wonly*mu) + randn(n,1) );
    Fnew(Fnew<-1000) = -1000; 
    Fnew(Fnew>1000) = 1000;
    
    if strcmp(model.diagonalSigmat, 'yes') == 1 
        Fnew((model.K*model.T+1):end) = 0;
    end
    
    hs = reshape(Fnew(1:model.K*model.T), model.T, model.K)';
    deltas = reshape(Fnew((model.K*model.T+1):end), model.T, model.tildeK)';
    omegas = (0.5*pi)*( (exp(deltas)-1)./(exp(deltas) + 1)); 
    
    %omegas = (0.5*pi)*tanh(0.5*deltas);
      
    % perform an evaluation of the log likelihood of the MSV model
    newloglik = 0;
    if Langevin == 1
        for t=1:model.T
          [tmp1, tmp2, tmpgradOmega, tmpgradHs] = msv_loglik(model.Ft(:,t), omegas(:,t), hs(:,t), model.Givset);
          gradDeltas(:,t) = tmpgradOmega'; 
          gradDeltas(:,t) = gradDeltas(:,t).*(pi*( exp(deltas(:,t))./( (1 + exp(deltas(:,t))).^2 ) ) ); 
          gradHs(:,t) = tmpgradHs'; 
          newloglik = newloglik + tmp1; 
        end 
    else     
        for t=1:model.T
          newloglik = newloglik + msv_loglik(model.Ft(:,t), omegas(:,t), hs(:,t), model.Givset);
        end
    end
    
    
    % Metropolis-Hastings to accept-reject the proposal
    corrFactor = 0;
    if Langevin == 1 
    %
        % new gradient 
        tmp = gradHs';  
        derFnew(1:model.K*model.T) = tmp(:); 
        tmp = gradDeltas';
        derFnew((model.K*model.T+1):end) = tmp(:);   
          
        corrFactor = - (Z' - F')*derF + (Z' - Fnew')*derFnew;
        corrFactor = corrFactor - (model.auxLikVar(1)/2)*(derFnew'*derFnew - derF'*derF);
    %    
    end
    
    [accept, uprob] = metropolisHastings(newloglik + corrFactor, oldloglik, 0, 0);  
    %
    if (it > BurnInIters) 
       acceptF = acceptF + accept;
    end   
    
    if accept == 1
         F = Fnew;
         %oldloglik = newloglik; 
         model.hs = hs;
         model.deltas = deltas;
         model.omegas = omegas;
      
         storeloglik =  newloglik;
         
         % log likelihood for phis which depend on Deltas and Hs !!!!!!
         %oldLogLphi = loggaussAR(model, model.K, model.tildeK); 
           
         ok = Lfree*(F - mu);
         oldLogLphi = - 0.5*(model.T*(model.K+model.tildeK))*log(2*pi) + sum(log(diag(Lfree))) - 0.5*(ok'*ok);
    end
    
    % STEP 4: Sample h_0s and delta_0s (Gibbs step)
    t1 = 1 - model.phi_h; 
    t2 = 1 - model.phi_h.^2;
    
    %model.h_0 = model.h_0./(t2  + (model.T-1)*(t1.^2)); 
    
    %%%%%%%%%%%%%%%%%%
    %model.h_0 = t2.*(model.hs(:, 1)')  + t1.* ( sum( model.hs(:, 2:end)  - repmat(model.phi_h', 1, model.T-1).*model.hs(:, 1:end-1), 2)' ); 
    %sh2 = model.sigma2_h./(t2  + (model.T-1)*(t1.^2)); 
    %model.h_0 = model.h_0./(t2  + (model.T-1)*(t1.^2))  + sqrt(sh2).*randn(size(sh2));
    %%%%%%%%%%%%%%%%%%
    model.h_0=(log(mean(model.Resid.^2,2)))';
    if strcmp(model.useFactorModel, 'yes') 
        model.h_0=(log(mean((model.Weights'*model.Resid).^2,2)))';
    end
    
    t1 = 1 - model.phi_delta; 
    t2 = 1 - model.phi_delta.^2;
    
    model.delta_0 = t2.*(model.deltas(:, 1)')  + t1.* ( sum( model.deltas(:, 2:end)  - repmat(model.phi_delta', 1, model.T-1).*model.deltas(:, 1:end-1), 2)' ); 
    %model.delta_0 = model.delta_0./(t2  + (model.T-1)*(t1.^2)); 
    sdelta2 = model.sigma2_delta./(t2  + (model.T-1)*(t1.^2));  
    model.delta_0 = model.delta_0./(t2  + (model.T-1)*(t1.^2)) + sqrt(sdelta2).*randn(size(sdelta2));

    % This affects the proposal over hs and deltas (the mean vector mu)
    tmp = repmat(model.h_0,model.T,1);
    mu = tmp(:);
    tmp = repmat(model.delta_0,model.T,1);
    mu = [mu; tmp(:)];
   
    % STEP 5: Sample sigma2_h and sigma2_delta (Gibbs step)
    newSigmar = 0.5*(model.T + model.priorSigma2_h.sigmar);  
    newSsigma = model.priorSigma2_h.Ssigma + (1 - model.phi_h.^2).*( (model.hs(:, 1)' - model.h_0).^2 ); 
    hhs = model.hs - repmat(model.h_0', 1, model.T);
    newSsigma = newSsigma  + sum( (hhs(:, 2:end)  - repmat(model.phi_h', 1, model.T-1).*hhs(:, 1:end-1)).^2 , 2)';
    newSsigma = 0.5*newSsigma;
    model.sigma2_h = 1./gamrnd(newSigmar, 1./newSsigma);
    %
    newSigmar = 0.5*(model.T + model.priorSigma2_delta.sigmar);  
    newSsigma = model.priorSigma2_delta.Ssigma + (1 - model.phi_delta.^2).*( (model.deltas(:, 1)' - model.delta_0).^2 ); 
    ddeltas = model.deltas - repmat(model.delta_0', 1, model.T);
    newSsigma = newSsigma  + sum( (ddeltas(:, 2:end)  - repmat(model.phi_delta', 1, model.T-1).*ddeltas(:, 1:end-1)).^2 , 2)';
    newSsigma = 0.5*newSsigma;
    model.sigma2_delta = 1./gamrnd(newSigmar, 1./newSsigma);
    
    
    %
    % this step affects the proposal over hs and deltas (W matrix)
    nonZeros = []; 
    for i=1:model.K
        %tmpM = tridiagAR(model.T, model.phi_h(i), model.sigma2_h(i));    
        %st = (i-1)*model.T + 1;
        %en = i*model.T;
        %W(st:en,st:en) = tmpM;  
        
        d = ((1+model.phi_h(i)^2)/model.sigma2_h(i))*ones(1,model.T);
        d(1) = 1/model.sigma2_h(i);
        d(end) = 1/model.sigma2_h(i);
        nonZeros = [nonZeros, d];
        offd = -model.phi_h(i)/model.sigma2_h(i);
        nonZeros = [nonZeros, offd*ones(1, 2*model.T - 2)];  
    end
    for j=1:model.tildeK
        %st = model.K*model.T + (j-1)*model.T + 1;
        %en = model.K*model.T + j*model.T;
        %tmpM = tridiagAR(model.T, model.phi_delta(j), model.sigma2_delta(j));
        %W(st:en,st:en) = tmpM;  
        
        d = ((1+model.phi_delta(j)^2)/model.sigma2_delta(j))*ones(1,model.T);
        d(1) = 1/model.sigma2_delta(j);
        d(end) = 1/model.sigma2_delta(j);
        nonZeros = [nonZeros, d];
        offd = -model.phi_delta(j)/model.sigma2_delta(j);
        nonZeros = [nonZeros, offd*ones(1, 2*model.T - 2)];  
    end
   
    W = sparse(iAll, jAll, nonZeros); 
    
    Lfree = jitterChol(W); 
    Wonly = W;
    W = W + diag(sparse(1./model.auxLikVar*ones(1,n))); 
    L = jitterChol(W)'; 
    
    
    % STEP 6:  SAMPLE ALL PHIS (M-H step)
    newtildePhi = randn(1,model.K + model.tildeK).*sqrt(PropDist.phi) + [model.tildephi_h, model.tildephi_delta];
    newtildePhi(newtildePhi<-10) = -10;
    newtildePhi(newtildePhi>10) = 10;
    newmodel = model;
    newmodel.tildephi_h = newtildePhi(1:model.K);
    newmodel.tildephi_delta = newtildePhi(model.K+1:end); 
    newmodel.phi_h = (exp(newmodel.tildephi_h) - 1)./(exp(newmodel.tildephi_h) + 1);
    newmodel.phi_delta = (exp(newmodel.tildephi_delta) - 1)./(exp(newmodel.tildephi_delta) + 1);

    nonZeros = []; 
    for i=1:newmodel.K
        %tmpM = tridiagAR(newmodel.T, newmodel.phi_h(i), newmodel.sigma2_h(i));    
        %st = (i-1)*newmodel.T + 1;
        %en = i*newmodel.T; 
        %Wtmp(st:en,st:en) = tmpM;
          
        d = ((1+newmodel.phi_h(i)^2)/newmodel.sigma2_h(i))*ones(1,newmodel.T);
        d(1) = 1/newmodel.sigma2_h(i);
        d(end) = 1/newmodel.sigma2_h(i);
        nonZeros = [nonZeros, d];
        offd = -newmodel.phi_h(i)/newmodel.sigma2_h(i);
        nonZeros = [nonZeros, offd*ones(1, 2*newmodel.T - 2)];  
    end
    for j=1:newmodel.tildeK
        %st = newmodel.K*newmodel.T + (j-1)*newmodel.T + 1;
        %en = newmodel.K*newmodel.T + j*newmodel.T;
        %tmpM = tridiagAR(newmodel.T, newmodel.phi_delta(j), newmodel.sigma2_delta(j));
        %Wtmp(st:en,st:en) = tmpM;
        
        d = ((1+newmodel.phi_delta(j)^2)/newmodel.sigma2_delta(j))*ones(1,newmodel.T);
        d(1) = 1/newmodel.sigma2_delta(j);
        d(end) = 1/newmodel.sigma2_delta(j);
        nonZeros = [nonZeros, d];
        offd = -newmodel.phi_delta(j)/newmodel.sigma2_delta(j);
        nonZeros = [nonZeros, offd*ones(1, 2*newmodel.T - 2)];  
    end
    
    Wtmp = sparse(iAll, jAll, nonZeros); 
    
    
    Lfreetmp = jitterChol(Wtmp); 
    ok = Lfreetmp*(F - mu);
    newLogLphi = - 0.5*(newmodel.T*(newmodel.K+newmodel.tildeK))*log(2*pi) + sum(log(diag(Lfreetmp))) - 0.5*(ok'*ok);
    
    % log likelihood for phis
    %newLogLphi = loggaussAR(newmodel, newmodel.K, newmodel.tildeK); 
    
    % log prior for phis
    if strcmp(model.exchangeablePriorphi, 'yes') 
       newPriorphi_h = logmarginalizedNormalGam(newmodel.tildephi_h, model.priorPhi_h.mu0, model.priorPhi_h.k0, model.priorPhi_h.alpha0, model.priorPhi_h.beta0); 
       newPriorphi_delta = logmarginalizedNormalGam(newmodel.tildephi_delta, model.priorPhi_delta.mu0, model.priorPhi_delta.k0, model.priorPhi_delta.alpha0, model.priorPhi_delta.beta0);
    else
       newPriorphi_h = logNormal(newmodel.tildephi_h, model.priorPhi_h.mu0, model.priorPhi_h.s2); 
       newPriorphi_delta = logNormal(newmodel.tildephi_delta, model.priorPhi_delta.mu0, model.priorPhi_delta.s2); 
    end 
    newPriorphi = newPriorphi_h + newPriorphi_delta;
     
    [accept, uprob] = metropolisHastings(newLogLphi + newPriorphi, oldLogLphi + oldPriorphi, 0, 0);
    if accept == 1
    %
       model = newmodel; 
       oldLogLphi = newLogLphi; 
       oldPriorphi = newPriorphi;
            
       % Update also the proposal for the hs and deltas which depends on the
       % Gauss AR prior 
       %for i=1:model.K
       %   tmpM = tridiagAR(model.T, model.phi_h(i), model.sigma2_h(i));    
       %   st = (i-1)*model.T + 1;
       %   en = i*model.T;
       %   W(st:en,st:en) = tmpM;
       %end
       %for j=1:model.tildeK
       %   st = model.K*model.T + (j-1)*model.T + 1;
       %   en = model.K*model.T + j*model.T;
       %   tmpM = tridiagAR(model.T, model.phi_delta(j), model.sigma2_delta(j));
       %   W(st:en,st:en) = tmpM;
       %end
           
       Lfree = Lfreetmp;
       % add the inverse of the auxiliary data variance in the diagonal
       Wonly = Wtmp;
       W = Wtmp + diag(sparse(1./model.auxLikVar*ones(1,n))); 
       L = jitterChol(W)';  
    %
    end
    if (it > BurnInIters) 
        acceptPhi = acceptPhi + accept;
    end
    
    % keep samples after burn in
    if (it > BurnInIters)  & (mod(it,StoreEvery) == 0)
    %
        cnt = cnt + 1; 
        samples.F(cnt,:) = F;
        samples.Ft(:,:,cnt) = model.Ft;
        samples.Phi_h(cnt,:) = model.phi_h;
        samples.Phi_delta(cnt,:) = model.phi_delta;  
        samples.h_0(cnt,:) = model.h_0;
        samples.delta_0(cnt,:) = model.delta_0;
        samples.sigma2_h(cnt,:) = model.sigma2_h;
        samples.sigma2_delta(cnt,:) = model.sigma2_delta;
        samples.BB(cnt,:,:) = BB;
        
        if model.VarConPara == 0 && model.VarTvAll == 0;
            samples.TvIntercept(cnt,:,:) = TvIntercept;
        end
        if model.VarConPara == 0 && strcmp(model.VarTvEstimation,'random_walk')
            %also store the draws of the volatility of the RW state
            %equations
            samples.VarQ(cnt,:,:)=model.VarQ0;
        end
        
        
        if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
        samples.sigma2(cnt,:) = model.sigma2;  
        else
        samples.sigma2(cnt) = model.sigma2;
        end
        samples.Weights(:,:,cnt) = model.Weights;
        if strcmp(model.useFactorModel, 'yes')
           if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
           tmp3 = bsxfun(@times, 1./model.sigma2', (model.Resid - (model.L.*model.Weights)*model.Ft).^2); 
           samples.LogL(cnt) = - 0.5*sum(num_OfNonNans.*log(2*pi*model.sigma2)) - 0.5*nansum(nansum( tmp3 ));   
           else
           samples.LogL(cnt) = - (0.5*num_OfNonNans)*log(2*pi*model.sigma2) - nansum(nansum( (model.Resid - (model.L.*model.Weights)*model.Ft).^2 ))/(2*model.sigma2);      
           end
        else
           samples.LogL(cnt) = storeloglik;
        end
        
    %
    end
    
    
end
%
%

model.hs = reshape(F(1:model.K*model.T), model.T, model.K)';
model.lambdas = exp(model.hs);
model.deltas = reshape(F((model.K*model.T+1):end), model.T, model.tildeK)';
model.omegas = (0.5*pi)*( (exp(model.deltas)-1)./(exp(model.deltas) + 1));


if strcmp(model.sampleFactorsByGibbs, 'yes') == 1
   accRates.Ft = 100*ones(model.T, 1);
else
   accRates.Ft = (acceptFt/Iters)*100;
end
accRates.Phi = (acceptPhi/Iters)*100; 
accRates.F = (acceptF/Iters)*100; 
