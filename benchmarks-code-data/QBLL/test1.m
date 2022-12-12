close all
clc
z = 1:99;
out=accuray_cal(z);
h=ape_cal(z,z);
a1=abs(z-z)*100;
a2=a1/(abs(z));