function [ X,y,Tact ] = getdatavar( insamp,p )
%get the data in VAR setup
[T,N]=size(insamp);

for l=1:p
    z=lag0(insamp,l);
    X(:,1+N*(l-1):l*N)=z(1+p:T,:);
end

y=insamp(1+p:T,:);
Tact=T-p;
X=[ones(Tact,1),X];

end

