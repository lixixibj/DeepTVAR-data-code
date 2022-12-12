function out=invpd(in)
temp=eye(size(in,2));
out=in\temp;