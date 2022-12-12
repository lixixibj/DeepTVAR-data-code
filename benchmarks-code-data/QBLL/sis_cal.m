%sis
%insample: (T,1)
%lower: (h,1)
%upper:(h,1)
%a=0.05
%outsample: (h,1)
function sis=sis_cal(insample,outsample,lower,upper,a,freq)
[T,m]=size(insample);
[h,n]=size(outsample);
l=T-freq;
error=zeros(l,1);
for j=(freq+1):T
    %Y.saved (m,l)
   error(j,1)=abs(insample(j,1)-insample((j-freq),1)) ;
end
masep=mean(error);
b=zeros(h,1);
for i=1:h
     U.subtract.L=upper(i,1)-lower(i,1);
    if outsample(i,1)<lower(i,1)
        r=(2/a)*(lower(i,1)-outsample(i,1));
    else
        r=0;
    end
  
    if outsample(i,1)>upper(i,1)
        q=(2/a)*(outsample(i,1)-upper(i,1));
    else
        q=0;  
    end
    
    b(i,1)= U.subtract.L+r+q;
end
sis=b/masep;
end










% 
% %sis
% freq=12;
% lower=forecast_output.lower_forecast(:,1);
% upper=forecast_output.upper_forecast(:,1);
% l=train_len-freq;
% error=zeros(l,1);
% for j=(freq+1):train_len
%     %Y.saved (m,l)
%    error(j,1)=abs(Y.saved(1,j)-Y.saved(1,(j-freq))) ;
% end
% masep=mean(error);
% 
% %model.actual: (m,h)
% a=0.05
% b=zeros(l,1)
% for i=1:h
%      U.subtract.L=upper(i,1)-lower(i,1);
%     if model.actual(1,i)<lower(i,1)
%         r=(2/a)*(lower(i,1)-model.actual(1,i));
%     else
%         r=0;
%    
%     if model.actual(1,i)>upper(i,1)
%         q=(2/a)*(model.actual(1,i)-upper(i,1));
%     else
%         q=0;
%     
%     b(i,1)= U.subtract.L+r+q;
% end
