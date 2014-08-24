close all;
clear all;

load gmm_data;

K = 3;

[R, C] = size(X);

miu = X(:, 1:3);

sig = zeros(R, R, K);

for i = 1 : K
    sig(:,:,i) = eye(2) * std(X(:));
end

X_rep = repmat(X, 3, 1);

prob = 1/K * ones(K, 1);

cost_prev = 0;

while(true)
    r = zeros(K, C);
    
    for i = 1 : C
        for j = 1 : K
            r(j, i) = prob(j) * mvnpdf(X(:, i), miu(:,j), sig(:,:,j));
        end
    end

    r = r ./ repmat(sum(r), 3, 1);

    N_k = sum(r, 2);

    for i = 1 : K
        miu(:, i) = (1/N_k(i)) * sum((repmat(r(i,:), 2, 1) .* X), 2);
    end

    for i = 1 : K
        inner_sum = zeros(R, R);
    
        for j = 1 : C
            inner_sum = inner_sum + r(i, j) * (X(:,j) - miu(:,i)) * (X(:,j) - miu(:,i))';
        end
    
        inner_sum  = inner_sum / N_k(i);
        
        sig(:,:,i) = inner_sum;
    end

    prob = sum(r,2) / C;

    cost = 0;

    for i = 1 : C
        inner_sum = 0;
        for j = 1 : K
            inner_sum = inner_sum + prob(j) * mvnpdf(X(:,i), miu(:,j), sig(:,:,j));
        end
        
        if(inner_sum == Inf)
            stop;
        end

        cost = cost + log(inner_sum);
    end
    
    diff = abs(cost - cost_prev);
    
    cost_prev = cost;
 
    diff
    
    if(diff < 0.00001)
        break
    end
end

