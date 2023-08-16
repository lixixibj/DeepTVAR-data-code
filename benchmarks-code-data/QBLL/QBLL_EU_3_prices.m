%forecast 3 series
%this code was written as part of Alan Turing Institute EPSRC grant EP/N510129/1
%the code estimates a TVP TV volatilty model using a variety of options
%the code allows for subsets of the intercepts, AR coefficients and
%covariance matrix to be drawn using either constant specification, random
%walk specification (as in Cogley and Sargent (2003, 2005) and Primiceri
%(2005) or using the QBLL approach of Petrova (2019)
% to select constant coefficients, select model.VarConPara=0 (then standard
% Minnesota priors and conditional conjugate posteriors are used for the parameters 
% for TV select model.VarConPara=1
% if coefficients are TV, select whether all (model.VarTvAll = 1;)
% or just intercepts are TV (model.VarTvAll = 0)
% then select estimation procedure for the TVP model.VarTvEstimation =
% 'random_walk' or 'kernel'
% for the covariance matrix there are four options
% use main_constant_primiceri_kernel_volatility to do:
% 1. constant volatility model.VarVolConPara = 1 uses standard Inverse Wishart priors
% and conditional posteriors
% 2. TV volatility model.VarVolConPara = 0 allows for model.VarVolEstimation =
% 'random_walk' - Primiceri (2005) specification with random walk log eigenvalues, 
% Gaussian mixtures as in Kim, Shepard and Chib (1998) and random walk lower unitriangular matrix
% or 3. 'kernel' implements the conditional TV quasi-posterior from Petrova (2019)
% this procedure does option 4. the dynamic eigenvalues, eigenvectors from 
% Dellaportas, Petrova, Plataniotis, Titsias 2019; this option requires a
% more elaborate MCMC with adaptation, so this option is implemented
% separately using main_msv_volatility.m option
% the code also allows for forecasting, which takes into account the TVP specifications
% to forecast the dynamic parameters clear

close all
clc

rng(2,'twister') % for reproducibility
[x,fval,exitflag,output] = ga(@rastriginsfcn, 2);

data = csvread('eu-3-prices-logged-for-matlab.csv');

num_of_forecast=20;
model.horizon  = 12; %forecast horizon
freq=12; %monthly dataset
[len,~] = size(data)
test_len=model.horizon+num_of_forecast-1
train_len=len-test_len

save_file_point='QBLL/point/';
save_file_lower='QBLL/lower/';
save_file_upper='QBLL/upper/';

K=3;
se_array=zeros(num_of_forecast, model.horizon, K);
ape_array=zeros(num_of_forecast, model.horizon, K);
is_array=zeros(num_of_forecast, model.horizon, K);
sis_array=zeros(num_of_forecast, model.horizon, K);

for num=1:num_of_forecast
num
b=num
e=num+train_len-1
%if forecasting
model.Y =(data(b:e,:))';
%Y_saved:(T,m)
Y_saved=(data(b:e,:));
model.actual = (data((e+1):(e+model.horizon),:))';
% model.Y =(data_q)';
[N,~] = size(model.Y);  
model.N = N;


mcmcoptions.adapt.T = 100;
mcmcoptions.adapt.Burnin = 0;
mcmcoptions.adapt.StoreEvery = 1;
mcmcoptions.adapt.disp = 1;
mcmcoptions.adapt.minAdapIters = 1; %100;
mcmcoptions.adapt.maxAdapIters = 2; %100;
mcmcoptions.train.T = 20; %250000;
mcmcoptions.train.Burnin = 0; %50000;
mcmcoptions.train.StoreEvery = 1; %20;

% START CREATING THE MODEL STRUCTURE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        %%%OPTIONS FOR THE VAR coefficients                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model.VarLag = 2; %lag order of the VAR
[ model.VarX,model.VarY,model.VarT ] = getdatavar( model.Y',model.VarLag );
model.Y = model.VarY'; 
model.T = model.VarT; % we lose some observations due to lagging
model.VarConPara = 0; % 1 if ALL parameters are constant, 0 if some TV
model.VarOverall_shrinkage = 1; %overall shrinkage for the priors of VAR parameters (this is lambda for Minnesota prior 
[model.VarPriorMean, model.VarPriorVar ] = get_Minnesota_prior( model.VarLag, model.VarOverall_shrinkage, model.VarY );
model.VarCheckStability=1; %check if eigenvalues are within unit circle
model.StabilityThreshold = 0.999; %upper bound on max abs eigenvalue
%model.StabilityThreshold = 2;
%FIT A PRELIMINARY VAR TO GET RESIDUALS
model.VarBB0=((model.VarX'*model.VarX)^(-1))*(model.VarX'*model.VarY);
resid=(model.VarY-model.VarX*model.VarBB0);
model.Resid=resid';
model.Bconstant=model.VarBB0(2:end,:); %remove the intercepts
             
if model.VarConPara == 0 %if time-varying AR parameters
% choose whether all parameters vary or just intercepts 
model.VarTvAll = 1; % 1 if ALL parameters are TV, 0 for just intercepts
%choose estimation method for the TV parameters; options are 'kernel' or
%'random_walk'
model.VarTvEstimation = 'kernel';
%model.VarTvEstimation = 'random_walk';
if strcmp(model.VarTvEstimation,'kernel')
model.VarKernelW=normker(model.T,sqrt(model.T)); %use Gaussian kernel for the application 
%with optimal bandwidth H=sqrt(T)
elseif strcmp(model.VarTvEstimation,'random_walk')
% set all hyperparametes here
sig0=(model.Resid*model.Resid')/model.T;
V0=kron(sig0,inv(model.VarX'*model.VarX));
%priors for the TV parameters
model.VarQ0=V0*model.T*1e-04; %priors for the variance of random walk equations
%this is super tight following the macro literature
constants = locateC( model.N, model.VarLag );
model.VarQ0(constants,constants)=10*model.VarQ0(constants,constants); %looser priors on the intercept RW variances
model.VarT0=30; %degrees of freedom for variance of random walk equations
model.VarP00=V0; %variance for the initial state vector in Kalman filter
model.VarB0=(model.Bconstant(:))'; %initial state vector
model.VarDiagStates=1; %0 allows for correlation in state equations

end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                          %%%OPTIONS FOR THE MSV parameters                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model.useFactorModel = 'no';
%
% -- Sample factors by Gibbs or not --
%    (this option has *no effect* if model.useFactorModel = 'no')
%      yes: it samples the factors by Gibbs (expensive, but exact)
%      no:  it samples the factors by auxiliary Langevin (faster)
model.sampleFactorsByGibbs = 'no';
%
% -- Diagonal Sigmat matrix or not --
%      yes: the Sigmat matrices are all diagonal (angles are zero)
%      no:  the Sigmat matrices have free form (angles are inferred)
model.diagonalSigmat = 'no';
%
% -- Exchangeable prior for the phis or just simple independent Gaussian with very
%    large variance
%    yes: exchangeable with normal-inverse gamma hyerprior
%    no:  just a simple broad Gaussian
model.exchangeablePriorphi = 'no';
% -- Single noise variance sigma2 ('no') in the final factro model
%    or ('yes') diagonal covariance (this option has been after the recent (2017) reviews)
%    yes:
%    no:
model.diagonalFinalLikelihoodNoise = 'yes';  %no if no extra noise 
%
% -- need_adapt
%
%    yes: call mcmcadapt
%    no:  do not call mcmcadapt

%distcomp.feature( 'LocalUseMpiexec', false );
%parpool
diagonalSigmat='no';

% randn('seed',0);
% randn('seed',0);

if strcmp(model.useFactorModel, 'no')
    K = N;
end

% create the Givens set
Givset = [];  % the indices
for i=1:K
    for j=i+1:K
        Givset = [Givset; i j];
    end
end
tildeK = size(Givset,1);

%model.T = T;
model.K = K;
model.Givset = Givset;
model.tildeK = size(Givset,1);


% PARAMETER INITIALIZATION FOR THE MCMC
model.deltas = zeros(model.tildeK, model.T);
model.omegas = (0.5*pi)*( (exp(model.deltas)-1)./(exp(model.deltas) + 1));
model.hs = repmat(zeros(K,1), 1, model.T);
model.lambdas = exp(model.hs);

L = ones(N,K);
L(1:K,1:K) = triu(ones(K,K))';

% HYPERPARAMETER INITIALIZATION FOR MCMC
ind =  ~isnan(model.Resid(:));
if strcmp(model.diagonalFinalLikelihoodNoise,'yes')
    model.sigma2 = var(model.Resid(ind))*ones(1,N); %0.01*var(model.Resid(ind))*ones(1,N);
else
    model.sigma2 = var(model.Resid(ind)); %0.01*var(model.Resid(ind));
end
model.L = L;
model.Weights = randn(N,K);
model.sigma2weights = 2;
model.Ft = zeros(model.K, model.T);
%model.FFt = Ft;
model.phi_h = 0.9*ones(1 ,K); %zeros(1,K);
model.tildephi_h = log((1 + model.phi_h)./(1 - model.phi_h));
model.h_0 = 5*ones(1,K);%zeros(1 ,K); 
model.sigma2_h = ones(1,K);%ones(1, K);
model.phi_delta = 0.9*ones(1,tildeK);%zeros(1, tildeK);
model.tildephi_delta = log((1 + model.phi_delta)./(1 - model.phi_delta));
model.delta_0 = zeros(1,tildeK);%zeros(1, tildeK);
model.sigma2_delta = ones(1, tildeK); %ones(1, tildeK);


% PRIOR OVER PHIS
if strcmp(model.exchangeablePriorphi, 'yes')
    model.priorPhi_h.type = 'logmarginalizedNormalGam';
    model.priorPhi_h.mu0 = 0;
    model.priorPhi_h.k0 = 1;
    model.priorPhi_h.alpha0 = 1;
    model.priorPhi_h.beta0 = 1;
    model.priorPhi_delta.type = 'logmarginalizedNormalGam';
    model.priorPhi_delta.mu0 = 0;
    model.priorPhi_delta.k0 = 1;
    model.priorPhi_delta.alpha0 = 1;
    model.priorPhi_delta.beta0 = 1;
else
    model.priorPhi_h.type = 'logNormal';
    model.priorPhi_h.mu0 = 3; %0;
    model.priorPhi_h.s2 = 5; %1000;
    model.priorPhi_delta.type = 'logNormal';
    model.priorPhi_delta.mu0 = 3; %0;
    model.priorPhi_delta.s2 = 5; %1000;
end
model.priorSigma2_h.sigmar = 10; %5
model.priorSigma2_h.Ssigma = 0.001;%0.01*model.priorSigma2_h.sigmar;
model.priorSigma2_delta.sigmar = 10; %5
model.priorSigma2_delta.Ssigma = 0.001;%0.01*model.priorSigma2_delta.sigmar;

% INVERSE GAMMA PRIOR OVER THE LIKELIHOOD NOISE VARIANCE
model.priorSigma2.type = 'invgamma';
model.priorSigma2.alpha0 = 5;%0.001;
model.priorSigma2.beta0 = 0.05;%0.001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                %%%RUN THE MCMC ADAPTATION                                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[model,~,PropDist] = katerina_msv_new_output_samples(mcmcoptions,'yes', model);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       %%%RUN THE MCMC                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[model,samples,PropDist] = katerina_msv_new_output_samples(mcmcoptions,'no',model,PropDist);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      %%%USE THE OUTPUT TO GENERATE FORECASTS                                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[forecast_output] = katerina_msv_forecast(samples,mcmcoptions,model);

%%%%%%%%%%%%%%%%%%%%%%%save to files%%%%%%%%%%%%%%%%
point_file=strcat(save_file_point,strcat(string(num),'.csv'))
csvwrite(point_file,(forecast_output.PointF)');
lower_file=strcat(save_file_lower,strcat(string(num),'.csv'))
csvwrite(lower_file,(forecast_output.lower_forecast)');
upper_file=strcat(save_file_upper,strcat(string(num),'.csv'))
csvwrite(upper_file,(forecast_output.upper_forecast)');

%calculate accuracy
for ts=1:K
    ape=ape_cal(model.actual(ts,:)',forecast_output.PointF(:,ts));
    ape_array(num,:,ts)=ape;
    se=se_cal(model.actual(ts,:)',forecast_output.PointF(:,ts));
    se_array(num,:,ts)=se;
    is=is_cal(model.actual(ts,:)',forecast_output.lower_forecast(:,ts),forecast_output.upper_forecast(:,ts),0.05);
    is_array(num,:,ts)=is;
    sis=sis_cal(Y_saved(:,ts),model.actual(ts,:)',forecast_output.lower_forecast(:,ts),forecast_output.upper_forecast(:,ts),0.05,freq);
    sis_array(num,:,ts)=sis;
end

end


%print accuracy
mean_se=mean(se_array,[1]);
mean_ape=mean(ape_array,[1]);
mean_is=mean(is_array,[1]);
mean_sis=mean(sis_array,[1]);
for ts=1:K
    disp('ts')
    ts
    disp('se:h1-12')
    mean_se(:,:,ts)
    disp('ape:h1-12')
    mean_ape(:,:,ts)
    disp('is:h1-12')
    mean_is(:,:,ts)
    disp('sis:h1-12')
    mean_sis(:,:,ts)
end

