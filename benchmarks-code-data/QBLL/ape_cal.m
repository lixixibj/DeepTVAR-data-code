% %ape
%outsample: (h,1)
%forecasts: (h,1)
%return ape:(h,1)
function ape=ape_cal(outsample,forecasts)
a=abs(outsample-forecasts)*100;
b=abs(outsample);
[h,m]=size(outsample);
ape=zeros(h,1);
for i=1:h
 ape(i,1)=a(i,1)/b(i,1);
end
end


% p=forecast_output.PointF;
% ape_array=zeros(2, h, m)
% a=abs(p(:,1)-p(:,2))*100;
% b=abs(p(:,1));
% [h,m]=size(p(:,1))
% 
% ape=zeros(h,1)
% for i=1:h
%    ape(i,1)=a(i,1)/b(i,1) 
% end