function [ draw ] = draw_fbeta( priorvar,priormean, X,y,p, Intercept,CheckStability,bound )
%this function draws contant beta
[T,N]=size(y');
priorprec0=priorvar^(-1);
s=0;
s2=0;
KK=size(X,2);

for t=1:T
xx=squeeze(X(:,:,t))'*squeeze(X(:,:,t));
s=xx+s;
xy=squeeze(X(:,:,t))'*y(:,t);
s2=xy+s2;
end
      
bayesprec=(kron(eye(N),priorprec0)+s);
bayesv=bayesprec^(-1);
BB=bayesv*(s2+(kron(eye(N),priorprec0)*priormean(:)));

check=0;
while check==0
drawbeta=BB + chol(bayesv)'*randn(KK,1); % Draw of beta 
draw = reshape(drawbeta,KK/N,N); % Draw of BETA

if CheckStability==1
if abs(max(eig(companion(draw',N,p, Intercept))))<bound
    check=1;
end
elseif CheckStability==0
    check=1;    
end
end

end