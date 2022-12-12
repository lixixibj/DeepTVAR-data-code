function [ a, sigma0 ] = get_priorsIW( p, Y )
%This is the IW priors, set using the automatic rule in Kadiyala and Karlsson (1997)
%written by Katerina
[T,N]=size(Y);

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
    
    a=max(N+2,N+2*8-T);
    sigma0=(a-N-1)*sigma_sq;
    sigma0=diag(sigma0);    
    
end