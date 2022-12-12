function [shat, sig] = initialize_KF_SV(Rlast, Vlast, prior)
% -------------------------------------------------------------------------
% Initialization of the KF in the step_sv
% -------------------------------------------------------------------------
% Rlast: rho of stochastic vol process (n by 1)
% Vlast: sig^2 of stochastic vol process (n by n)
% where n is number of stochastic vol processes
% -------------------------------------------------------------------------
% NOTE
% - works for independent stoachastic processes
% - works for NO LONG RUN MEAN
% -------------------------------------------------------------------------
% stationary case
%sig  = diag(Vlast./(1-Rlast.^2));
%shat = ones(size(Rlast));

% nonstatinoary case ...
n=size(Vlast,1);
sig = zeros(n);
shat = zeros(n,1);
if prior.unitroot == 0
    % stationary case
    sig  = diag(Vlast./(1-Rlast.^2));
    shat = ones(size(Rlast));
elseif prior.unitroot == 1
    % nonstatinoary case ...
    n    = size(Vlast,1);
    sig  = eye(n);%zeros(n,n);
    shat = -1*ones(n,1);%zeros(n,1);
end


