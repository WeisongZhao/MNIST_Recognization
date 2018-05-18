function [ output_args ] = test( W,B )
%TEST Summary of this function goes here
%   Detailed explanation goes here
layer_num = size(W,2);
test_image = loadMNISTImages('test-images');
test_label = loadMNISTLabels('test-labels');

for i = 1:10000
    
    label_tmp = zeros(10,1);
    label_tmp(test_label(i)+1) = 1;
    test_images{i} = {test_image(:,i) label_tmp};
end
equal = 0;
for i = 1:10000
    [A,Z] = feedforward(test_images{i}{1},W,B);
    [junk,index] = max(A{layer_num});
    if test_images{i}{2}(index) == 1
        equal = equal + 1;
        
    end
end
fprintf('%d/10000\n',equal);
