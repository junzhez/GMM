close all;
clear all;

load gmm_data;

K = 3;

[R, C] = size(X);

miu = X(:, 1:3);

X_rep = repmat(X, 3, 1);

cost_prev = 0;

while(true)

    miu_re = reshape(miu, [R * K, 1]);

    dist = (X_rep - repmat(miu_re, 1, C)).^2;

    dist = reshape(dist, [R, K * C]);

    dist = sum(dist);

    dist = reshape(dist, [K, C]);

    [val, ind] = min(dist);

    r = zeros(C, K);

    for i = 1 : C
        r(i, ind(i)) = 1;
    end

    cost = sum(sum(r' .* dist));

    for i = 1 : K
        miu(:,i) = sum((repmat(r(:, i)', 2, 1) .* X),2) / sum(r(:,i));
    end

    abs(cost - cost_prev)
    
    if(abs(cost - cost_prev) < 1)
        break;
    end
    
    cost_prev = cost;
end

