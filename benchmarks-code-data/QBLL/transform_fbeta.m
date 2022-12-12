function [e]=transform_fbeta(X,y,BB)

%transform data with f beta
    e=y-X*BB;

end

