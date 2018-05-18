function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

for k=1:1:60000
    [a,b]=size(images(:,:,k));
    
for i=1:1:a
    for j=1:1:b   
        if  images(i,j,k)~=0    
            up=i;           
            break       
        end
    end
end
for i=a:-1:1
    for j=1:1:b       
        if  images(i,j,k)~=0            
            down=i;
            break
        end        
    end
end
for j=1:1:b
    for i=1:1:a       
        if  images(i,j,k)~=0            
            left=j;
            break
        end        
    end
end
for j=b:-1:1
    for i=1:1:a       
        if  images(i,j,k)~=0            
            right=j;
            break
        end        
    end
end
images1=images(down:up,right:left,k);
images1=imresize(images1,[size(images, 1) size(images, 2)]);
images1=reshape(images1, size(images, 1) * size(images, 2), 1);
images2(:,k)=images1;
end

% Reshape to #pixels x #examples
% images = reshape(images2, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images =abs(double(images2)/255) ;

end