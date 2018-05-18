function [ nabla_b,nabla_w ] = backprop( A,W,Z,Y )
%网页链接中的4个公式，计算权值和偏置的修正量
%BACKPROP Summary of this function goes here
%   Detailed explanation goes here
    layer_num = size(A,2);
    DELTA{layer_num} = (A{layer_num} - Y).*sigmoid_prime(Z{layer_num});
    nabla_b{layer_num} = DELTA{layer_num};
    nabla_w{layer_num} = DELTA{layer_num}*A{layer_num-1}';
    for j = layer_num-1:2
        DELTA{j} = (W{j+1}'*DELTA{j+1}).*sigmoid_prime(Z{j});
        nabla_b{j} = DELTA{j};
        nabla_w{j} = DELTA{j} * A{j-1}';
    end

end

