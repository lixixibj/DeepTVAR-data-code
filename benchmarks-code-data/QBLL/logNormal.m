function out = logNormal(x, mu, s2)
%

out = - 0.5*log(2*pi*s2) - (0.5/s2)*((x - mu).^2);

out = sum(out);
