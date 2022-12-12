function [ drawbeta ] = draw_TVbeta( priorvar,priormean, Wei, X,y,p,CheckStability,bound)
%this function draws TVP beta, conditional on fixed sigma
[T,N]=size(y');
kk=size(X,2);
drawbeta=zeros(kk,T);
priorprec0=priorvar^(-1);
pp=kron(eye(N),priorprec0);
pm=priormean(:);
for t=1:T
    w=Wei(t,:);
s=0;
s2=0;
for tt=1:T
xx=squeeze(X(:,:,tt))'*w(tt)*squeeze(X(:,:,tt));
s=xx+s;
xy=squeeze(X(:,:,tt))'*w(tt)*y(:,tt);
s2=xy+s2;
end
      
bayesprec=pp+s;
bayesv=bayesprec^(-1);
BB=bayesv*(s2+pp*pm);

check=0;
while check==0
drawbeta(:,t)=BB + chol(bayesv)'*randn(kk,1); % Draw of beta 
draw = reshape(drawbeta(:,t),kk/N,N); 

if CheckStability==1
if abs(max(eig(companion(draw',N,p, 1))))<bound
    check=1;
end
elseif CheckStability==0
    check=1;
end
end

end