function [ N ] = Gaussian(x, miu, sig)    
    N = (x - miu)' * pinv(sig) * (x - miu);
    
    D = size(sig, 1);
    
    N = (2 * pi)^(-D/2) * (det(sig))^(-1/2) * exp(N);
end