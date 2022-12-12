function [Y,XX]=transform_tvsigma(X,y,Sigma)
% Sigma is NxNxT history of TV volatility
%Transforms the X and y, to get a model with variance covariance identity
[T,N]=size(y);

Y=zeros(N,T);
XX=zeros(N,N*size(X,2),T);
for t=1:T
    Y(:,t)=(chol(squeeze(Sigma(:,:,t))))^(-1)*y(t,:)';
    %XX(:,:,t)=chol((squeeze(Sigma(:,:,t))))^(-1)*kron(eye(N),X(t,:));
    XX(:,:,t)=kron(chol((squeeze(Sigma(:,:,t))))^(-1),X(t,:));
end


