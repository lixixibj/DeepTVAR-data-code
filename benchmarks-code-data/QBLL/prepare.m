function [ X ] = prepare( data,L )
X=[];
for j=1:L
X=[X lag0(data,j) ];
end
X=[X ones(rows(X),1)];
end

