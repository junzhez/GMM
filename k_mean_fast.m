close all;
clear all;

load gmm_data;

K = 3;

[R, C] = size(X);

miu = X(:, 1:3);

X_rep = repmat(X, 3, 1);

cost_prev = 0;

learning_rate = 0.001;

while(true)

    miu_re = reshape(miu, [R * K, 1]);

    dist = (X_rep - repmat(miu_re, 1, C)).^2;

    dist = reshape(dist, [R, K * C]);

    dist = sum(dist);

    dist = reshape(dist, [K, C]);

    [val, ind] = min(dist);

    r = zeros(C, K);
    
    % the fast kmean, miu_k_new = miu_k_old + learning_rate * 
    % (x_n - miu_k_old)
    for i = 1 : C
        r(i, ind(i)) = 1;
        miu(:,ind(i)) = miu(:,ind(i)) + learning_rate*(X(:,i) - miu(:,ind(i)));
    end

    cost = sum(sum(r' .* dist));

    abs(cost - cost_prev)
    
    if(abs(cost - cost_prev) < 0.001)
        break;
    end
    
    cost_prev = cost;
end

plot(X(1,:), X(2,:), '.');
hold on;
plot(miu(1,:), miu(2,:), 'r.');

