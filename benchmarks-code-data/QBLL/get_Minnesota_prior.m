function [ priormean, priorvar ] = get_Minnesota_prior( p, overall_shrinkage, Y )
%This sets a Minnesota-type prior for the AR coefficients of the VAR using roughly Banbura et al.

%written by Katerina 12/09/2019
%data are in growth rates, so prior mean is zero not 1
[T,N]=size(Y);
K=N*p+1;
priormean=zeros(K,N);

priorvar=zeros(K,1);  %this has a kronerker structure

sigma_sq = zeros(N,1); % vector to store residual variances
    for i = 1:N
        % Create lags of dependent variable in i-th equation
        Ylag_i = mlag2(Y(:,i),p);
        Ylag_i = Ylag_i(p+1:T,:);
        % Dependent variable in i-th equation
        Y_i = Y(p+1:T,i);
        % OLS estimates of i-th equation
        alpha_i = ((Ylag_i'*Ylag_i)^(-1))*(Ylag_i'*Y_i);
        sigma_sq(i,1) = (1./(T-p+1))*(Y_i - Ylag_i*alpha_i)'*(Y_i - Ylag_i*alpha_i);
    end
s=sigma_sq.^(-1);
    for ll=1:p
        priorvar(2+N*(ll-1):1+N*ll)=(overall_shrinkage^2)*s/(ll^2);  
    end
    priorvar(1)=10^2; %for the constant on top; leave the prior very loose
    priorvar=diag(priorvar);
        
end

