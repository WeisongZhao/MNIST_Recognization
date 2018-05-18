function [ W,B ] = train( epoches, eta, layer )
%三个输入分别为 epoches为训练次数，eta表示学习速率，layer神经网络结构，输入示例：[m,n]=train(10, 3, [784, 30, 10]);
image = loadMNISTImages('train-images'); % 加载样本图像共60000，size(image)=784*60000
label = loadMNISTLabels('train-labels');%加载样本图像对应标签
for i = 1:60000
    label_tmp = zeros(10,1);
    label_tmp(label(i)+1) = 1;
    images{i} = {image(:,i) label_tmp};%定义Cell变量把样本与标签对应地存储起来
end 
%TRAIN Summary of this function goes here
%   Detailed explanation goes here
layer_num = size(layer,2);%确定神经网络结构的层数
%size(layer)=1,3; size(layer,2)=3;
batch_num = 10;

for i = 2:layer_num%随机初始化权值w与偏置b，赋值为【-1，1】之间的随机数
    W{i} = [];
    B{i} = [];
    W{i} = randn(layer(i),layer(i-1));
    %layer= [784, 30, 10]，则layer(2)=30，layer(3)=10；
    B{i} = randn(layer(i),1);
    %以上两行代码分别给出了由第2层到第3层，第3层在反向到第2层的权值和偏置
end 

for i = 1:epoches %训练次数循环
    r = randperm(size(images,2));%随机打乱60000个训练样本的标签，然后再进行训练
    images = images(:,r);
    for j = 1:batch_num:59991%以10个训练样本为一组，来修正权值w和偏置b
        for k = 2:layer_num%只有从第2层以后才有权值w和偏置b
            NABLA_B{k} = zeros(layer(k),1);%偏置初始化
            NABLA_W{k} = zeros(layer(k),layer(k-1));%权值初始化
        end
        cur_batch = images(:,j:j+batch_num-1);%从image中取出10个图像*标签
        for k = 1:batch_num% 对10个图像进行循环
            [A, Z]= feedforward(cur_batch{k}{1},W,B);%A包含各层的输出
            [nabla_b,nabla_w] = backprop(A,W,Z, cur_batch{k}{2});%计算权值和偏置的修正量
            
            for m = 2:layer_num
                NABLA_B{m} = NABLA_B{m} + nabla_b{m};
                NABLA_W{m} = NABLA_W{m} + nabla_w{m};
                %上两行代码：对层数循环，然后对偏置和权值求和
            end
        end
        %fprintf('%f %d %f %f %f\n',A{3}(2),j,B{2}(1),NABLA_B{2}(1), eta.*NABLA_B{2}(1)./batch_num);
        for k = 2:layer_num
            B{k} = B{k} - eta.*NABLA_B{k}./batch_num;
            W{k} = W{k} - eta.*NABLA_W{k}./batch_num;
        end
    end
%  test(W,B);
end
end

