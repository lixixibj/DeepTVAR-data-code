function out = iwpq(v,ixpx)

k=size(ixpx,1);
z=zeros(v,k);
mu=zeros(k,1);
for i=1:v
    z(i,:)=(cholx(ixpx)'*randn(k,1))';
end
out=inv(z'*z);

