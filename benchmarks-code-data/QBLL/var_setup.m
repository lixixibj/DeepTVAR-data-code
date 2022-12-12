function [ y,X,Tact ] = var_setup( insamp, p )
%takes insample and lag length and gives var matrices y, X

[Tsam,N]=size(insamp);

for l=1:p
    z=lag0(insamp,l);
    X(:,1+N*(l-1):l*N)=z(1+p:Tsam,:);
end

y=insamp(1+p:Tsam,:);
Tact=Tsam-p;
X=[ones(Tact,1),X];

end

