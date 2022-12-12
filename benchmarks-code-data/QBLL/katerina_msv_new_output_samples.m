function [model,samples,PropDist] = katerina_msv_new_output_samples(mcmcoptions,need_adapt,model,PropDist)
% Y(N,T) = data
% K = number of latent factors
%
% -- Factor model or just simple MSV --
%    (missing values will be sampled by Langevin for simple MSV, while for the factor are marginalized out)
%    yes: factor MSV
%    no:  simple MSV
%model.useFactorModel = 'no';
%
% -- Sample factors by Gibbs or not --
%    (this option has *no effect* if model.useFactorModel = 'no')
%      yes: it samples the factors by Gibbs (expensive, but exact)
%      no:  it samples the factors by auxiliary Langevin (faster)
%model.sampleFactorsByGibbs = 'no';
%
% -- Diagonal Sigmat matrix or not --
%      yes: the Sigmat matrices are all diagonal (angles are zero)
%      no:  the Sigmat matrices have free form (angles are inferred)
%model.diagonalSigmat = 'yes';
%
% -- Exchangeable prior for the phis or just simple independent Gaussian with very
%    large variance
%    yes: exchangeable with normal-inverse gamma hyerprior
%    no:  just a simple broad Gaussian
%model.exchangeablePriorphi = 'no';
% -- Single noise variance sigma2 ('no') in the final factro model
%    or ('yes') diagonal covariance (this option has been after the recent (2017) reviews)
%    yes:
%    no:
%model.diagonalFinalLikelihoodNoise = 'yes';  %no if no extra noise 
%
% -- need_adapt
%
%    yes: call mcmcadapt
%    no:  do not call mcmcadapt

K = model.K;
T  = model.T;
tildeK = model.tildeK;
N = model.N;
Givset = model.Givset;

% HERE WE RUN THE MCMC ALGORITHM FIRST TO ADAPT THE PROPOSAL AND THEN TO
% COLLECT THE SAMPLES
% the option Langevin=1 is not implemented yet
Langevin = 1;
%Langevin

if strcmp(need_adapt, 'yes')
    
    tic;
    [model, PropDist, samples, accRates] = mcmcAdapt(model, mcmcoptions.adapt, Langevin);
    % training/sample collection phase
    elapsedAdapt=toc;
    
else
    
    tic;
    [model, samples, accRates] = mcmcTrainVarF(model, PropDist, mcmcoptions.train, Langevin);
    elapsedTrain = toc;
    
    % PLACE THE SAMPLES IN THE ORIGINAL FORM
    m = mean(samples.F);
    sd = sqrt(var(samples.F));
    samples.hs = zeros(size(model.hs,1), size(model.hs,2), size(samples.F,1));
    samples.deltas = zeros(size(model.deltas,1), size(model.deltas,2), size(samples.F,1));
    for s=1:size(samples.F,1)
        samples.hs(:,:,s) = reshape(samples.F(s,1:K*T), T, K)';
        samples.deltas(:,:,s) = reshape(samples.F(s,(K*T+1):end), T, tildeK)';
    end
    samples.mean_hs = reshape(m(1:K*T), T, K)';
    samples.sd_hs = reshape(sd(1:K*T), T, K)';
    samples.mean_deltas = reshape(m((K*T+1):end), T, tildeK)';
    samples.sd_deltas = reshape(sd((K*T+1):end), T, tildeK)';
    
    % COMPUTE MONTE CARLO AVERAGES FOR THE COVARIANCES ACROSS TIME
    
    estimSigma = zeros(N,N,T);
    estimSigmaFactors = zeros(K,K,T);
    S = size(samples.F,1);
    for s = 1:S
        omegas = (0.5*pi)*( (exp( samples.deltas(:,:,s) )-1)./(exp(  samples.deltas(:,:,s) ) + 1));
        lambdas = exp(samples.hs(:,:,s));
        LW = model.L.*samples.Weights(:,:,s);
        for t=1:T
            G = eye(K);
            for k=1:size(Givset,1)
                Gtmp = givensmat(omegas(k, t), K, Givset(k,1), Givset(k,2));
                G = Gtmp*G;
            end
            estimSigmaFactors(:,:,t) = estimSigmaFactors(:,:,t) +  G*diag(lambdas(:,t))*G';
            estimSigma(:,:,t) = estimSigma(:,:,t) + LW*(G*diag(lambdas(:,t))*G')*LW';
        end
    end
    estimSigma = estimSigma/S;
    estimSigmaFactors = estimSigmaFactors/S;
    
    fullestimSigma = zeros(N,N,T);

    for t=1:T
        fullestimSigma(:,:,t) = estimSigma(:,:,t) + diag(mean(samples.sigma2,1));
    end
    
    samples.FutureSigma = LW*(G*diag(lambdas(:,T))*G')*LW' + diag(mean(samples.sigma2,1));
    samples.estimSigma = estimSigma;
    samples.estimSigmaFactors = estimSigmaFactors;
    samples.fullestimSigma = fullestimSigma;
    samples.accRates =  accRates;
    
end
