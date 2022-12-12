function [logscore,pit] = lnsc(density,actual)

%This code is written by Katerina Petrova, 27/08/15

%Evaluates the log predictive score at a grid 
grid=linspace(min(density)-5*std(density),max(density)+5*std(density),10000); %grid for kernel evaluating
lss=ksdensity(density,grid); %kernel density 

%INPUTS
%grid is 1xN grid
%density is 1xN values of density evaluated at each grid point
%actual is a scalar realised value

%OUTPUTS
% logscore is a scalar log predictive score, ie the height of the pdf
% evaluated at actual
% index gives the exact location on the grid where the actual is

[~,N]=size(grid);
for ii=1:N
    if grid(ii)<=actual&&grid(ii+1)>=actual
        pdf=0.5*(lss(ii))+0.5*(lss(ii+1));
        index=ii;
    end
end
logscore=log(pdf);
%computation of PITs
pit=sum(lss(:,(1:index)))/sum(lss); %cdf at the actual