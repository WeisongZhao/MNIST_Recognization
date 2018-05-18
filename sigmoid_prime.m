function [ prime ] = sigmoid_prime( z )
%SIGMOID_PRIME Summary of this function goes here
%   Detailed explanation goes here
prime = sigmoid(z).*(1-sigmoid(z));

end

