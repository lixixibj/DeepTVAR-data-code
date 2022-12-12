function [samples] = GibbsTrain_vol(trainOps,model)
BurnInIters = trainOps.Burnin; 
Iters = trainOps.T; 
StoreEvery = trainOps.StoreEvery;
newresiduals = zeros(model.T,model.N);
cnt = 0;

constant_conjugacy = (model.VarConPara == 1)&&(model.VarVolConPara == 1); 
%in this case we have N IW close form, no need for Gibbs  

if constant_conjugacy
    %compute the analyic posterior moments of N-IW posterior
    
priorprec0=model.VarPriorVar^(-1);
bayesprec=(priorprec0+model.VarX'*model.VarX);
bayesv=bayesprec^(-1);
BMean=bayesv*(model.VarX'*model.VarY+priorprec0*model.VarPriorMean);
bayesalpha=model.VarVolPriorA+model.T;
g1=model.VarPriorMean'*priorprec0*model.VarPriorMean;
g2=model.VarY'*model.VarY;
g3=BMean'*bayesprec*BMean;
bayesgamma=model.VarVolPriorB+g1+g2-g3;
bayesgamma=0.5*bayesgamma+0.5*bayesgamma'; %it is symmetric but with inverses, matlab sometimes gets stuck
    
end

kernel_conjugacy   = (model.VarConPara == 0)&&(model.VarTvAll == 1)&&...
    strcmp(model.VarTvEstimation,'kernel')&& (model.VarVolConPara == 0)&&...
    strcmp(model.VarVolEstimation,'kernel'); 
%in this case we have TVP N IW close form, no need for Gibbs 
if kernel_conjugacy
    %compute the TVP analyic posterior moments of N-IW posterior, as in
    %Petrova 2019 JoE
    for tt=1:model.T
    w = model.VarKernelW(tt,:); %corresponding row of kernel weights matrix
    priorprec0 = model.VarPriorVar^(-1);
    bayesprec = (priorprec0+model.VarX'*(diag(w))*model.VarX);
    kernel_pv(:,:,tt) = bayesprec^(-1);
    kernel_pmean(:,:,tt) = squeeze(kernel_pv(:,:,tt))*((model.VarX'*diag(w))*model.VarY+priorprec0*model.VarPriorMean);
    kernel_bayesalpha(tt) = model.VarVolPriorA+sum(w);
    g1 = model.VarPriorMean'*priorprec0*model.VarPriorMean;
    g2 = model.VarY'*diag(w)*model.VarY;
    g3 = squeeze(kernel_pmean(:,:,tt))'*bayesprec*squeeze(kernel_pmean(:,:,tt));
    gamma = model.VarVolPriorB+g1+g2-g3;
    kernel_bayesgamma(:,:,tt) = 0.5*gamma+0.5*gamma';
    
    end
end

estimSigma=model.Sigma0;  %starting value for Sigma(t)
   
for it = 1:(BurnInIters + Iters) 
    it
%WITHIN EACH GIBBS ITERATION

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                             Step 1                                            %
%%%                                     Draw the VAR coefficients                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                           Constant Parameters                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         if model.VarConPara == 1&&~constant_conjugacy % constant parameters draw without conjugacy
         [Y,XX]=transform_tvsigma(model.VarX,model.VarY,estimSigma);

         BB = draw_fbeta( model.VarPriorVar,model.VarPriorMean, XX,Y,model.VarLag,1,model.VarCheckStability,model.StabilityThreshold);
         residuals=transform_fbeta(model.VarX,model.VarY,BB);
         
         model.Resid=residuals';

         else
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                           TV PARAMETERS                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

             if model.VarConPara == 0 && model.VarTvAll == 1 % ALL TV parameters
             

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%                             %KERNEL                            %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                 if strcmp(model.VarTvEstimation,'kernel')&&~kernel_conjugacy  % kernel draw without conjugacy
                 [Y,XX]=transform_tvsigma(model.VarX,model.VarY,estimSigma);
                 BB = draw_TVbeta( model.VarPriorVar,model.VarPriorMean,model.VarKernelW,XX,Y,model.VarLag,model.VarCheckStability,model.StabilityThreshold);
                 residuals=transform_tvbeta(model.VarX,model.VarY,BB);
                 model.Resid=residuals';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%                             %RANDOM WALK                         %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                 elseif strcmp(model.VarTvEstimation,'random_walk')
                 %draw the betas
                 if model.VarDiagStates==1
                 model.VarQ0=diag(diag(model.VarQ0));
                 end
                 [BB,residuals,~,~]=carterkohn_coef(model.VarY,model.VarX,model.VarQ0,estimSigma,...
                     (model.VarBB0(:))',model.VarP00,model.VarLag,model.VarCheckStability,model.StabilityThreshold,10);
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%               %TV INTERCEPTS ONLY, KERNEL                        %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                 
             elseif model.VarConPara == 0 && model.VarTvAll == 0 % ONLY intecepts vary
             VarXnoC=model.VarX(:,2:end);
             Ytilda=transform_fbeta(VarXnoC,model.VarY,model.Bconstant);
             [Y,XX]=transform_tvsigma(ones(model.T, 1),Ytilda,estimSigma);

             if strcmp(model.VarTvEstimation,'kernel')
             %draw the TV intercepts
             TvIntercept = draw_TVbeta( model.VarPriorVar(1,1),model.VarPriorMean(1,:),model.VarKernelW,XX,Y,model.VarLag,0,0);
             %TvIntercept = draw_TVintercept( model.VarPriorVar(1,1),model.VarPriorMean(1,:),model.VarKernelW,Ytilda');
             
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%            %TV INTERCEPTS ONLY, RANDOM WALK                      %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

             elseif strcmp(model.VarTvEstimation,'random_walk')
             %draw the TV intercepts with KF
             locationC = locateC( model.N, model.VarLag );
             if model.VarDiagStates==1
                 model.VarQ0=diag(diag(model.VarQ0));
             end
             [TvIntercept,~,~,~]=carterkohn_coef(Ytilda,ones(model.T,1),model.VarQ0(locationC,locationC),...
                 estimSigma,model.VarB0(locationC),model.VarP00(locationC,locationC),model.VarLag,0,0,10);
             %draw the Qs
                 errorQ=diff(TvIntercept);
                 scaleQ=(errorQ'*errorQ)+model.VarQ0(locationC,locationC)+0.01;
                 if model.VarDiagStates==0
                 model.VarQ0(locationC,locationC)=iwpq(model.T-1+model.VarT0, invpd(scaleQ));
                 else
                 model.VarQ0(locationC,locationC)=iwpq(model.T-1+model.VarT0, invpd(diag(diag(scaleQ))));    
                 end
                 TvIntercept=TvIntercept';

             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%                      %DRAW THE CONSTANT BETAS                    %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
             VarYnoC = model.VarY-TvIntercept';
             [Y,XX]=transform_tvsigma(model.VarX(:,2:end),VarYnoC,estimSigma);
             model.Bconstant = draw_fbeta( model.VarPriorVar(2:end,2:end),model.VarPriorMean(2:end,:), XX,Y,model.VarLag, 0, model.VarCheckStability,model.StabilityThreshold);
             BB=model.Bconstant;
             residuals=(model.VarY-TvIntercept'-model.VarX(:,2:end)*model.Bconstant);
             model.Resid=residuals';
             end
             
         end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                 Step 2                                        %
%%%                                  Draw the VOLATILITY MATRIX                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%                      %DRAW CONSTANT VOLATILITY                   %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if model.VarVolConPara ==1&&~constant_conjugacy % constant volatility draw without conjugacy
    bayesalpha = model.VarVolPriorA + model.T; %posterior DoF
    bayesgamma = model.VarVolPriorB + model.Resid*model.Resid'; %posterior Scale Matrix
    estimSigma = iwishrnd(bayesgamma,bayesalpha); %draw a matrix sigma from IW
    estimSigma = repmat(estimSigma,1,1,model.T);
    
elseif model.VarVolConPara == 0  % tv volatility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%                             %KERNEL                            %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(model.VarVolEstimation,'kernel')&&~kernel_conjugacy  % kernel draw without conjugacy
    for tt=1:model.T
 bayesalpha = model.VarVolPriorA + sum(model.VarVolKernelW(tt,:)); %TV posterior DoF
 bayesgamma = model.VarVolPriorA + (model.Resid*diag(model.VarVolKernelW(tt,:))*model.Resid'); %TV Scale Matrix
 estimSigma(:,:,tt) = iwishrnd(bayesgamma,bayesalpha); %draw a matrix sigma from IW
    end   
end

if strcmp(model.VarVolEstimation,'random_walk')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                  Draw the LOWER TRIANGLAR MATRIX                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

[model.SVolamatx, model.SVolDD, model.QA] = draw_A_QA(residuals,exp(model.SVolh),model.SVolC0,model.SVolPC0,...
    model.SVolDD,model.SVolD0, model.SVolT0,model.SVolamatDiag);

%compute new residuals consitional on TV draw of lower triangular A(t)
for tt = 1:model.T
newresiduals(tt,:) = chofac(model.N,model.SVolamatx(tt,:)')*residuals(tt,:)';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                               Draw the VOLATILITIES and MIXING WEIGHTS                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

[model.SVolh,~,model.Vlast,~,~,~,~,~] = step_sv_primiceri(newresiduals,model.SVolh,ones(model.N,1),...
    model.Vlast,zeros(model.N,1),model.priorSV);

%compute the implied TV VOl
for t=1:model.T
            A = invpd(chofac(model.N,model.SVolamatx(t,:)'));
            H = diag((exp(2*model.SVolh(t,:)))');
            estimSigma(:,:,t) = A*H*A';
end  
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     CONSTANT CONJUGACY                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
if constant_conjugacy
    %this is a special case textbook constant parameters Inverse Wishart posterior, no need
    %for Gibbs
    
    check=0;
while check==0
Sigma = iwishrnd(bayesgamma,bayesalpha); %draw a matrix sigma from IW
nu=randn(model.N*model.VarLag+1,model.N);
BB=(BMean+chol(bayesv)'*nu*(chol(Sigma)));
if model.VarCheckStability==1
    [Ficom,~]=companion(BB',model.N,model.VarLag, 1);
if max(abs(eig(Ficom)))<model.StabilityThreshold %check if draw is stationary
     check=1; 
end
elseif model.VarCheckStability==0
    check=1;
end
end
estimSigma = repmat (Sigma,1,1,model.T);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                      KERNEL CONJUGACY                                      %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if kernel_conjugacy
    %this TVP Inverse Wishart posterior, as in Petrova (2019) JoE. no need
    %for Gibbs
for tt=1:model.T 
    check=0;
while check==0
estimSigma(:,:,tt) = iwishrnd(squeeze(kernel_bayesgamma(:,:,tt)),kernel_bayesalpha(tt)); %draw a matrix sigma from IW
nu=randn(model.N*model.VarLag+1,model.N);
%Bdraw=(squeeze(kernel_pmean(:,:,tt))+chol(squeeze(kernel_pv(:,:,tt)))'*nu*(chol(squeeze(estimSigma(:,:,tt)))))';
Bdraw=(squeeze(kernel_pmean(:,:,tt))+chol(squeeze(kernel_pv(:,:,tt)))'*nu*(chol(squeeze(estimSigma(:,:,tt)))));

if model.VarCheckStability==1
    [Ficom,~]=companion(Bdraw',model.N,model.VarLag, 1);
if max(abs(eig(Ficom)))<model.StabilityThreshold %check if draw is stationary
     check=1; 
end
elseif model.VarCheckStability==0
    check=1;
end

end 
BB(:,tt) = Bdraw(:);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                 STORAGE                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

if (it > BurnInIters)  && (mod(it,StoreEvery) == 0)
    %
        cnt = cnt + 1; 
        samples.BB(cnt,:,:) = BB;
         if model.VarConPara == 0 && strcmp(model.VarTvEstimation,'random_walk')
            %also store the draws of the volatility of the RW state
            %equations
            samples.VarQ(cnt,:,:)=model.VarQ0;
         end
        
        samples.Sigma (cnt,:,:,:) = estimSigma;
        
        if model.VarVolConPara == 0 && strcmp(model.VarVolEstimation,'random_walk')
        samples.SVolamatx (cnt,:,:) = model.SVolamatx;
        samples.QA (cnt,:,:) = model.QA;
        samples.Vlast (cnt,:) = model.Vlast;
        samples.SVolh (cnt,:,:) = model.SVolh;
        end
        
        if model.VarConPara == 0 && model.VarTvAll == 0
            samples.TvIntercept(cnt,:,:) = TvIntercept;
        end
       
 end
 
end
end

