function [ drawbeta ] = draw_TVintercept( priorvar,priormean, Wei,y)
%this function draws TVP beta, conditional on fixed sigma
[T,N]=size(y');
drawbeta=zeros(N,T);
priorprec0=priorvar^(-1);
pp=kron(eye(N),priorprec0);
pm=priormean(:);
for t=1:T
    w=Wei(t,:);
s2=y*w';
s=sum(w);
bayesprec=pp+eye(N)*(s);
bayesv=bayesprec^(-1);
BB=bayesv*(s2+pp*pm);

drawbeta(:,t)=BB + chol(bayesv)'*randn(N,1); % Draw of beta 

end
end