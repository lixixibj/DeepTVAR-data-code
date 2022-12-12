function [ locationC ] = locateC( N, p )
%gets indices of intercepts in vec(B') where B is a N*p+1 x N vector of
%parameters
locationC=zeros(1,N);
for nc=1:N
             locationC(nc) = 1+(nc-1)*(N*p+1);
end

end

