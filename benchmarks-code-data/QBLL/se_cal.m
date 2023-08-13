% %ape
%outsample: (h,1)
%forecasts: (h,1)
%return se:(h,1)
function se=se_cal(outsample,forecasts)
se=(outsample-forecasts).^ 2;
end

