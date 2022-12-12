function [H,R,V,C,s,sigt,h_aT,h_PT]=step_sv_primiceri(e,Hlast,Rlast,Vlast,Clast,prior)
% -------------------------------------------------------------------------
% function r=step_sv(e,Hlast,Vlast,in_shat,priorV,priorT,a)
% -------------------------------------------------------------------------
% * This code makes a draw from the conditional posterior of
% the volatility states and the variance of the innovations
% to the log volatility
% * Full description is in Justiniano and Primiceri's AER paper (Appendix)
% * THIS IS PORTABLE VERSION: ONLY FUNCTION YOU NEED TO PROVIDE IS 
%   -> initialize_KF_SV
% -------------------------------------------------------------------------
% REQUIRED USER SPECIFIED FUNCTION
% - USER MUST SPECIFY initialize_KF_SV.m (initialization for filtering)
% -------------------------------------------------------------------------
% MODEL
% -------------------------------------------------------------------------
% e = sig_t * eps_t
% log(sig_{t}) = RHO*log(sig_{t-1}) + nu_t
% -------------------------------------------------------------------------
% Approximated Model
% -------------------------------------------------------------------------
% e_tilde = 2*eye(n)*h_t + ee,      ee ~ log(chi2)
% h_{t} = RHO*h_{t-1} + nu_{t},     nu ~ N(0,s^2)
% where
%       n = # of stochastic vol
%       RHO = diag([rho_1 rho_2 ... rho_n])
%       e_tilde = log((eta)^2 + cba)) where cba = 0.001
% -------------------------------------------------------------------------
% Algorithm (Step2~Step4 in AER paper)
% -------------------------------------------------------------------------
% Step 3: Draw inidcator (indicator for mixture of normals)
% Step 2: Draw Stochastic Volatility (h_{t})
% Step 4: Draw parameters (RHO, s) -> Rlast, Vlast
% -------------------------------------------------------------------------
% Input and Output
% -------------------------------------------------------------------------
% INPUT
%     e = observed innovations of the model with stochastic volatility (nobs by n)
%     Hlast = last draw of the time varying (log) standard deviations (nobs by n)
%     Rlast = last draw of the rho of the stochastic volatility (n by 1)
%     Vlast = last draw of the variance of the innovations to
%             the log standard deviations (n by 1)
%     prior.R1 = prior for R (mean)
%     prior.R2 = prior for R (variance cov matrix)
%     prior.V  = IG prior for V (n by 1)
%     prior.T  = degrees of freedom of the prior for V (n by 1) (0 if want flat prior) 
%     prior.unitroot = indicator (1: unitroot model, 0: cov. stationary model)
% 
% following three are not used anymore
%     in_shat = vector with guess
%     ind_corr = indicator variable: ind_corr=0 if stochastic volatilities
%                of different shocks are independent; ind_corr=1 otherwise.
%     scale_svprop = Proposal scalar for the initial state of the filter
% OUTPUT
%     H,R,V : new draws
%     s     : new draws of the indicators for mixture normal
%     sigt  : exp(H)
% -------------------------------------------------------------------------
% Notice that here the log volatilities are assumed to follow random walk processes.
% This can be easily changed.
% setting parameters for the mixture of normals to approximate the logX^2
% Create cks_mm cks_vv global cksmm cksvv ckspr
% -------------------------------------------------------------------------
% This function calls
% toolbox             :  \stats\stats\normpdf.m
% current dir         :  cks_mats
% current dir         :  gibbs_linreg_beta
% current dir         :  gibbs_linreg_sig2
% current dir         :  initialize_KF_SV
% current dir         :  kback_KF_SV
% current dir         :  kfilter_KF_SV
% current dir         :  selQ
% -------------------------------------------------------------------------
% Code from Dongho Song(Penn) - Original Code is by Primiceri and Justiniano
% Modified by Minchul Shin (Penn) 2012/04/01
% -------------------------------------------------------------------------

[T,n]=size(e);
[cksmm, cksvv, ckspr]=cks_mats(T);

% e=log(e.^2+.001);
e=log(e.^2+.00000001);
% -------------------------------------------------------------------------
% STEP3: drawing the new indicator variables for the mixture of normals
% -------------------------------------------------------------------------
s=zeros(T,n);
for i=1:n
    Q=ckspr.*normpdf(repmat(e(:,i),1,7),2*repmat(Hlast(:,i),1,7)+cksmm, cksvv );
    s(:,i)=selQ(Q);
end
v2=cksvv(1,:);
v2=v2.*v2;
 m=cksmm(1,:);

% -------------------------------------------------------------------------
% STEP2: drawing h_{t}
% -------------------------------------------------------------------------
% forward recursion
SHAT=zeros(T,n);
 SIG=zeros(T,n);
% initialize recursion
% sig=diag(scale_svprop);
% shat=in_shat(:);
[shat, sig] = initialize_KF_SV_diebold(Rlast, Vlast,prior);

% if prior.unitroot==0
%     sig  = diag(Vlast./(1-Rlast.^2));
%     shat = (1-Rlast).\Clast;
% elseif prior.unitroot == 1
%     sig = diag((1-prior.R1 .^2).\(prior.V ./ (prior.T-1))*3^2); %we initialize 3 times of variance of ht under stationary prior
%     shat = zeros(n,1);
% end

% t=1;
for t=1:T
    % filter has no intercept
    [shat,sig]= kfilter_KF_SV_setQone(e(t,:)'-m(s(t,:))',2*eye(n),diag(Rlast),shat,sig,diag(v2(s(t,:))),diag(Vlast), Clast);
        
    SHAT(t,:) = shat';
     SIG(t,:) = diag(sig)';
end

h_aT = shat;
h_PT = diag(sig);

% first draws and backward recursion
H=zeros(T,n);
% H(T,:)=mvnrnd(shat,sig,1);

H(T,:) = (shat + sqrt(diag(sig)).*randn(n,1))'; % instead of mvnrnd
for t=T-1:-1:1
    [btTp,StTp]=kback_KF_SV_setQone(SHAT(t,:)',diag(SIG(t,:)),H(t+1,:)',diag(Rlast),diag(Vlast), Clast);
    %H(t,:)=mvnrnd(btTp,StTp,1);
    H(t,:) = (btTp + sqrt(diag(StTp)).*randn(n,1))'; % instead of mvnrnd
end

% -------------------------------------------------------------------------
% STEP4: draw parameters
% -------------------------------------------------------------------------
% draws for R (linear reg of bet given sigma^2 (Vlast))
YY = H(2:end,:);
XX = H(1:end-1,:);
X0 = ones(size(H,1)-1,1); %intercept

 V = zeros(n,1);

if prior.unitroot == 1
    
    % unit-root (force coefficient to one)
    R = ones(n,1);
    C = zeros(n,1);
    % gibbs sampler of sig2 given rho=1
    for i = 1:1:n
        V(i,1) = gibbs_linreg_sig2(YY(:,i), XX(:,i), prior.T(i), prior.V(i), 1);
    end
    
else
    % H(t) has intercept: cov. stationary case (acc,rej sampling)
    R = zeros(n,1);
    C = zeros(n,1);
    for i = 1:1:n
        
        % gibbs sampler of rho given sig2
        covstat = 0;
        while covstat == 0
            temp_R = gibbs_linreg_beta(YY(:,i), [X0, XX(:,i)], [prior.C1(i); prior.R1(i)], diag([prior.C2(i);prior.R2(i)]), Vlast(i,1));
            if abs(temp_R(2)) < 1
                covstat = 1;
            end
        end
        C(i,1) = temp_R(1);
        R(i,1) = temp_R(2);
        
        % gibbs sampler of sig2 given rho
        V(i,1) = gibbs_linreg_sig2(YY(:,i), [X0, XX(:,i)], prior.T(i), prior.V(i), temp_R);
    end
    
%     % cov. stationary case (acc,rej sampling)
%     R = zeros(n,1);
%     for i = 1:1:n
%         
%         % gibbs sampler of rho given sig2
%         covstat = 0;
%         while covstat == 0
%             temp_R = gibbs_linreg_beta(YY(:,i), XX(:,i), prior.R1(i), prior.R2(i), Vlast(i,1));
%             if abs(temp_R) < 1
%                 covstat = 1;
%             end
%         end
%         R(i,1) = temp_R;
%         
%         % gibbs sampler of sig2 given rho
%         V(i,1) = gibbs_linreg_sig2(YY(:,i), XX(:,i), prior.T(i), prior.V(i), temp_R);
%     end
    
end

% -------------------------------------------------------------------------
% draws for the variances of the innovations to log(se)
% JP's code
% -------------------------------------------------------------------------
%     YY = H(2:end,:);
%     XX = H(1:end,:);
%     EE = YY - XX*diag(R); %residual
% 
%     VH = diag(diag( EE'*EE*(T-1)+diag(prior.V .*prior.T)));
%  shock = mvnrnd(zeros(n,1),eye(n)/VH, T-1+priorT);
%      V = diag(1./diag(shock'*shock));
% Previous code: a bit weird since cov means the series
% % draws for the variances of the innovations to log(se)
%     VH = diag(diag(cov(H(2:end,:)-H(1:end-1,:))*(T-1)+priorV*priorT));
%  shock = mvnrnd(zeros(n,1),eye(n)/VH,T-1+priorT);
%      V = diag(1./diag(shock'*shock));
% -------------------------------------------------------------------------

% Qt=exp(H);
% sigt=Qt';
sigt = exp(H);


% end of function
end




% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function [cksmm,cksvv,ckspr]=cks_mats(T)
% function [cksmm,cksvv,ckspr]=cks_mats(T);
% 
% Function creates the matrices cksmm, cksvv, ckspr 
% of T rows, wehere each row of 

% cksmm:   cotains the means of the 7 element mixture used
%           in approximating the SV model in Chib, Kim and Shephard 
% cksvv:   standard deviations of the 7 element mixture 
% 
% ckspr:   weights of the mixture 
pr=[.0073 .10556 .00002 .04395 .34001 .24566 .25750]';
m =[-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819]'-1.2704;
v2=sqrt( [5.79596 2.61369 5.1795 .16735 .64009 .34023 1.26261]' );
cksmm=repmat(m',T,1); 
cksvv=repmat(v2',T,1); 
ckspr=repmat(pr',T,1); 
end

% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function [colindic,Qind,u,Q]=selQ(Q) 
% function [colindic,Qind,u,Q]=selQ(Q); 
% 
% Given an r x c matrix Q
%
% Normalize (inside the code) Q such thate columns add up to 1 
% For each row i, select the column j 
% such that given a set of random draws u (rx1) from a uniform [0,1]
% u(i) <= Q(i,j) and u(i) > Q(i-1,j)
% 
% Output 
% =======
% indic     position indicator of the selected element in Q 
% Qind      matrix of dimension rxc such that the j=th column is equal to 1
%           in the it-th row and zero otherwise 
% u         vector of uniform draws 
% Q         renormalized Q probabilities 
%
% Alejandro Justinino  ajustiniano@imf.org 
% Created:          1/21/05 
% Last Modified:    2/3 /05  Removed  eps as a lower bound 
%                   4/10/07  Use to compute P(S) 
%
% Remarks:          Normalization done inside the code! 
%                   Lower bound in u is eps = 2.2204e-016
% To extract the associated multinomial probabilities 
% A=Q.*Qind; A( A~= 0 ) 
% Note: this will operate by columns and will NOT respect the time i.e. row
% 
% dimension !
% =================================================================
[r,c]=size(Q); 
Q=Q./repmat(sum(Q,2),1,c); 
Qcdf=cumsum(Q,2); 
Qind=zeros(r,c); 
u=unifrnd(0,100000,r,1)/100000; 
%u=unidrnd(100000,r,1)/100000;
umat=repmat(u,1,c); 
indic=  (umat <= Qcdf) & ( umat > [zeros(r,1) Qcdf(:,1:end-1)])  ; 
Qind(indic) =1; 
[colindic,junk]=find(Qind'==1); 
end
  
% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function betdraw = gibbs_linreg_beta(y,x,bet0,V0, sig2)
% -------------------------------------------------------------------------
% linear regression
% - gibbs for beta condi on sig^2
% -------------------------------------------------------------------------
% MODEL
% y = x*bet + sig*eps, eps ~ N(0,1)
% -------------------------------------------------------------------------
% INPUT
% -------------------------------------------------------------------------
% y, x     : data
% bet0, V0 : prior
% sig2     : var of eps, sig^2
% -------------------------------------------------------------------------

% posterior moments (bet|sig^2)
  V1 = (V0^(-1) + sig2^(-1)*(x'*x))^(-1);
bet1 = V1*(V0^(-1)*bet0 + sig2^(-1)*(x'*y));

% posterior draws
betdraw = bet1 + chol(V1)'*randn(size(bet1));
end

% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function sig2draw = gibbs_linreg_sig2(y, x, T, V, bet)
% -------------------------------------------------------------------------
% linear regression
% - gibbs for sig2 condi on bet
% -------------------------------------------------------------------------
% MODEL
% y = x*bet + sig*eps, eps ~ N(0,1)
% -------------------------------------------------------------------------
% INPUT
% -------------------------------------------------------------------------
% y, x     : data
% T, V     : prior T is shape and V is scale parameter, mean(sig2) = V/(T-1)
% bet      : previous draw
% -------------------------------------------------------------------------
% info
[nobs, npara] = size(x);

% posterior moments (sig^2|bet)
  EE = (y-x*bet);
  T1 = nobs/2 + T ;
  V1 = EE'*EE/2 + V;
  
% posterior draws sig2draw ~ IG(T1, V1) ~ [G(T1, V1^(-1))]^(-1)
sig2draw = gamrnd(T1, V1^(-1))^(-1);

end

% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function [shatnew,signew,loglh]=kfilter_KF_SV_setQone(y,H,F,shat,sig,R,Q,C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL
% y(t)=H*s(t)+e(t)
% s(t)=F*s(t-1)+v(t)
% V(e(t))=R
% V(v(t))=Q
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  omega = F*sig*F'+Q;
  sigma = H*omega*H'+R;
      k = omega*H'/sigma; 
   sfor = C+F*shat; 
   ferr = y-H*sfor; 
shatnew = sfor+k*ferr;
 signew = omega-k*H*omega;
 loglh = -( log(det(sigma)) + (ferr'/sigma)*ferr) / 2; 

% When there is intercept CC
% function [shatnew,signew,loglh]=kfilter_gio_ar(y,H,F,C,shat,sig,R,Q)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % MODEL
% % y(t)=H*s(t)+e(t)
% % s(t)=C+F*s(t-1)+v(t)
% % V(e(t))=R
% % V(v(t))=Q
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  omega = F*sig*F'+Q;
%   sigma = H*omega*H'+R;
%       k = omega*H'/sigma; 
%    sfor = C + F*shat; 
%    ferr = y-H*sfor; 
% shatnew = sfor+k*ferr;
%  signew = omega-k*H*omega;
%  loglh = -( log(det(sigma)) + (ferr'/sigma)*ferr) / 2; 
end

% -------------------------------------------------------------------------
% fucntions 
% -------------------------------------------------------------------------
function [btTp,StTp]=kback_KF_SV_setQone(Att, Ptt, bf, TT, Q, C)
% -------------------------------------------------------------------------
% Backward recursion (Carter and Korn)
% -------------------------------------------------------------------------
% MODEL
% y = ZZ*a + e        , e~N(0,H)
% a = TT*a(-1) + n    , n~N(0,Q)
% -------------------------------------------------------------------------
Phat_f = TT*Ptt*TT' + Q;        
inv_f  = inv(Phat_f);        
   cfe = bf - TT*Att - C;

btTp = (Att + Ptt*TT'*inv_f*cfe);
StTp = Ptt - Ptt*TT'*inv_f*TT*Ptt;


% When there is intercept CC
% function [btTp,StTp]=kback_KF_SV(Att, Ptt, bf, TT, CC, Q)
% -------------------------------------------------------------------------
% Backward recursion (Carter and Korn)
% -------------------------------------------------------------------------
% MODEL
% y = ZZ*a + e             , e~N(0,H)
% a = CC + TT*a(-1) + n    , n~N(0,Q)
% -------------------------------------------------------------------------
% Phat_f = TT*Ptt*TT' + Q;        
% inv_f  = inv(Phat_f);        
%    cfe = bf - TT*Att - CC;
% 
% btTp = (Att + Ptt*TT'*inv_f*cfe);
% StTp = Ptt - Ptt*TT'*inv_f*TT*Ptt;
end











