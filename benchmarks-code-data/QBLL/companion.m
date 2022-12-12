function [Ficom,mu]=companion(Fi1,n,l, Intercept)
%companion matrix
%Fi1=reshape(ss,n,n*l+1);
if Intercept==1
    mu=Fi1(:,1);
Ficom(1:n,:)=(Fi1(:,2:n*l+1));
elseif Intercept==0
    mu=[];
Ficom(1:n,:) = Fi1;   
end

for zz=2:l
     Ficom(1+n*(zz-1):zz*n,1+n*(zz-2):n*(zz-1))=eye(n);
 end

end

