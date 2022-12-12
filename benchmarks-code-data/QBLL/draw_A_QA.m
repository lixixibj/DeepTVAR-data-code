function [amat, DD, QA] = draw_A_QA(residuals,hlast,C0,PC0,DD,D0,T0,SVolamatDiag)
%makes a from lower tringular A matrix 
[T,N]=size(residuals);
QA=zeros((N*(N-1))/2,(N*(N-1))/2);
j=1;
for i=2:N
    ytemp=residuals(:,i);
    xtemp=residuals(:,1:i-1)*-1;
    ytemp=(ytemp)./(hlast(:,i));
    xtemp=(xtemp)./repmat((hlast(:,i)),1,cols(xtemp));
    a0=C0(i,1:i-1);
    pa=PC0*diag(abs(a0));
    Qa=DD{i-1};
    [a1,~]=carterkohn1(a0,pa,ones(T+1,1),Qa,ytemp,xtemp);
    amat(:,j:j+cols(a1)-1)=a1;
    %sample Qa
    a1errors=diff(a1);
    scaleD1=(a1errors'*a1errors)+D0{i-1};
    
    
    DD{i-1}=iwpq(T+T0,invpd(scaleD1)); %draw from inverse Wishart
    if SVolamatDiag==1
    DD{i-1} = diag(diag(DD{i-1}));
    end
    QA(j:j+cols(a1)-1,j:j+cols(a1)-1)=DD{i-1};
    j=j+cols(a1);
end
end

