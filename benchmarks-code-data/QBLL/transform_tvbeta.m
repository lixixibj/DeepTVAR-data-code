function [e]=transform_tvbeta(X,y,drawbeta)

%transform data with tv beta
[T,N]=size(y);
K=size(drawbeta,1)/N;

e=zeros(T,N);
for t=1:T
    B=reshape(drawbeta(:,t),K,N);
    e(t,:)=y(t,:)-X(t,:)*B;
end
end

