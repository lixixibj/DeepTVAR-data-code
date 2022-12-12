function [forecast_output] = katerina_msv_forecast(samples,mcmcoptions,model)
ForNSim=(mcmcoptions.train.T)/mcmcoptions.train.StoreEvery; %the number of MCMC and predictive density draws 
Density=zeros(model.horizon, model.N, ForNSim);
estimatedSigema=zeros(model.K, model.K, model.horizon, ForNSim);
for it = 1:ForNSim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                             Step 1                                            %
%%%              forecast the TV parameters forecast h steps ahead                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     if model.VarConPara == 1 % constant parameters
         beta(:,:,1:model.horizon)=repmat(squeeze(samples.BB(it,:,:)),1,1,model.horizon); %each draw from posterior is kept constant over forecast horizon

         elseif model.VarConPara == 0 && model.VarTvAll == 1 % ALL TV parameters
                 if strcmp(model.VarTvEstimation,'kernel')
                     
                 beta(:,:,1:model.horizon) = repmat(reshape(squeeze(samples.BB(it,:,model.T)),model.N*model.VarLag+1,model.N),1,1,model.horizon);
                     
                 %each draw from posterior is kept constant over forecast
                 %horizon, since kernel has no law of motion for B


                 elseif strcmp(model.VarTvEstimation,'random_walk')
                     
                 VarQ=squeeze(samples.VarQ(it,:,:));
                 BBLast=squeeze(samples.BB(it,model.T,:));
                 %project the random walk into the future
                 for hh=1:model.horizon
                 BBLast = BBLast+(mvnrnd(zeros(size(VarQ,1),1),VarQ))';
                 beta(:,:,hh) = reshape(BBLast,model.N*model.VarLag+1,model.N);
                 end 
                 end

                 
             elseif model.VarConPara == 0 && model.VarTvAll == 0; % ONLY intecepts vary

             if strcmp(model.VarTvEstimation,'kernel')
             TvIntercept(:,1:model.horizon) = repmat(squeeze(samples.TvIntercept(it,:,model.T))',1,model.horizon);
              %each draw from posterior is kept constant over forecast
              %horizon, since kernel has no law of motion for mu
              
             elseif strcmp(model.VarTvEstimation,'random_walk')
             %project the random walk into the future
                locationC = locateC( model.N, model.VarLag );
                VarQ=squeeze(samples.VarQ(it,locationC,locationC));
                TvInterceptLast=(squeeze(samples.TvIntercept(it,:,model.T)))';
                %project the random walk into the future
                 for hh=1:model.horizon
                 TvIntercept(:,hh) = TvInterceptLast+(mvnrnd(zeros(size(VarQ,1),1),VarQ))';
                 TvInterceptLast     = TvIntercept(:,hh);
                 end
             
             
             end
             % and the corresponding constant betas for tv intercepts
             BB(:,:,1:model.horizon)=repmat(squeeze(samples.BB(it,:,:)),1,1,model.horizon); %each draw from posterior is kept constant over forecast horizon
             beta(1,:,:)=TvIntercept;
             beta(2:model.N*model.VarLag+1,:,:)=BB;

     end
                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                 Step 2                                        %
%%%                                  Forecast Volatility Matrix                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

%         F            =      samples.F(it,:);
%         Ft           =      samples.Ft(:,:,it);      
%     hs = reshape(Fnew(1:model.K*model.T), model.T, model.K)';
%     deltas = reshape(Fnew((model.K*model.T+1):end), model.T, model.tildeK)';
%     omegas = (0.5*pi)*( (exp(deltas)-1)./(exp(deltas) + 1)); 
    
%project Sigma into the future 

        phi_h        =      samples.Phi_h(it,:);
        phi_delta    =      samples.Phi_delta(it,:);  
        h_0          =      samples.h_0(it,:);
        delta_0      =      samples.delta_0(it,:);
        sigma2_h     =      samples.sigma2_h(it,:);
        sigma2_delta =      samples.sigma2_delta(it,:);
        %Weights      =      samples.Weights(:,:,it);
        LW           =      model.L.*samples.Weights(:,:,it);

        hLast        =      samples.hs(:,model.T,it);
        deltaLast    =      samples.deltas(:,model.T,it);
        estimSigma   =      zeros (model.K, model.K, model.horizon);
        for t=1:model.horizon
        hs           =      (eye(size(hLast,1))-diag(phi_h))*h_0' + diag(phi_h) * hLast + (mvnrnd(zeros(size(hLast,1),1), diag(sigma2_h)))';
        deltas       =      (eye(size(deltaLast,1))-diag(phi_delta))*delta_0' + diag(phi_delta) * deltaLast + (mvnrnd(zeros(size(deltaLast,1),1), diag(sigma2_delta)))';
        hLast        =      hs;
        deltaLast    =      deltas;
        omegas       = (0.5*pi)*( (exp( deltas )-1)./(exp(  deltas ) + 1));
        lambdas      = exp(hs);
        ss2          = samples.sigma2(it,:);
            G = eye(model.K);
            for k=1:size(model.Givset,1)
                
                Gtmp = givensmat(omegas(k), model.K, model.Givset(k,1), model.Givset(k,2));
                G = Gtmp*G;
            end
            %estimSigmaFactors(:,:,t) = estimSigmaFactors(:,:,t) +  G*diag(lambdas(:,t))*G';
            %estimated Sigma of VAR would be W*EstSigma*W' where W are the
            %sample.Weights
        %    estimSigma(:,:,t) =  LW*(G*diag(lambdas)*G')*LW';
             estimSigma(:,:,t) =  (G*diag(lambdas)*G');   
        end
        estimatedSigema(:,:,:,it)=estimSigma;
        
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                     FORECAST THE SERIES                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YLast=model.VarY(model.T,:);
XLast=model.VarX(model.T,2:end); %get rid of constant
for t=1:model.horizon
            XLast=[YLast,XLast(1:model.N*(model.VarLag-1))];
            YLast = (beta(:,:,t)'*[1,XLast]')'+(LW*(mvnrnd(zeros(model.K,1),squeeze(estimSigma(:,:,t))))')'+(diag(ss2)*(mvnrnd(zeros(model.N,1),eye(model.N)))')';
            Density(t,:,it)=YLast;
end         

end

%***********************************************************************
% Compute point forecasts, bias, MSE, LPS, PIT and save results
%***********************************************************************
%estimSigma
estimatedSigema = mean(estimatedSigema,4);

PointF=mean(Density,3);
p = [0.025 0.975];
forecast_output.lower_forecast=quantile(Density,0.025,3);
forecast_output.upper_forecast=quantile(Density,0.975,3);
quantile(Density,0.5,3);
forecast_output.PointF=PointF;
forecast_output.estimatedSigema=estimatedSigema;
forecast_output.beta=beta;
forecast_output.Ferror=PointF-model.actual';

%Density
for hh=1:model.horizon
for var=1:model.N
 %log score/pits
[forecast_output.LS(hh,var),forecast_output.PIT(hh,var)]=lnsc(squeeze(Density(hh,var,:)),model.actual(var,hh));
end
end

end

