function [draw]=transform_tvsigma_drawfbeta(X,y,Sigma,priorprec0,priormean,p)

%Transforms the X and y, to get a model with variance covariance identity
[T,N]=size(y);
s1=0;
s2=0;
for t=1:T
    S=(squeeze(Sigma(:,:,t)))^(-1);
    s1=s1+kron(S,X(t,:)'*X(t,:));
    s2=s2+S*y(t,:)'*X(t,:);
end
invv=kron(eye(N),priorprec0);
bayesprec=invv+s1;
bayesv=bayesprec^(-1);
BB=bayesv*(s2(:)+invv*priormean(:));

drawbeta=BB + chol(bayesv)'*randn((N*p+1)*N,1); % Draw of beta 
draw = reshape(drawbeta,N*p+1,N); % Draw of BETA

