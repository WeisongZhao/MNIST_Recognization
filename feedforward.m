function [ A,Z ] = feedforward( a,W,B )
%FEEDFORWARD Summary of this function goes here
%   Detailed explanation goes here
layer_num = size(B,2);
A{1} = a;
for i = 2:layer_num
    Z{i} = W{i}*A{i-1}+B{i};
    A{i} = sigmoid(Z{i});%sigmoid( z )激活函数，为S型
end
end

