

%Neural Networks Codes will be run on this part

% tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;
clear all
pic=imread('self_writing\3.jpg');

pic1=imread('8.png');
pic1=rgb2gray(pic1);
% pic=255-pic;
[a,b]=size(pic1);
for i=1:1:a
    for j=1:1:b
        if  pic1(i,j)==0
            up=i;
            break
        end
    end
end
for i=a:-1:1
    for j=1:1:b
        if  pic1(i,j)==0
            down=i;
            break
        end
    end
end
for j=1:1:b
    for i=1:1:a
        if  pic1(i,j)==0
            left=j;
            break
        end
    end
end
for j=b:-1:1
    for i=1:1:a
        if  pic1(i,j)==0
            right=j;
            break
        end
    end
end
pic=pic1(down:up,right:left);
imshow(pic)
pic=imresize(pic,[28 28]);
% size(pic);
pic1=1-double(reshape(pic,784,1))/255;
[m,n]=train(10, 3, [784, 30, 10]);
[A,Z]=feedforward(pic1,m,n);
aa=A{3}(:,1);
i = find(aa==max(aa));
i=i-1;
fprintf('The Handwritten Number is %d\n',i);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc

clc
clear all

