% function are called with two types
% either cnn_cifar('coarse') or cnn_cifar('fine')
% coarse will classify the image into 20 catagories
% fine will classify the image into 100 catagories
function cnn_cifar(type, varargin)

if ~(strcmp(type, 'fine') || strcmp(type, 'coarse')) 
    error('The argument has to be either fine or coarse');
end

% record the time
tic
%% --------------------------------------------------------------------
%                                                         Set parameters
% --------------------------------------------------------------------
%
% data directory
opts.dataDir = fullfile('cifar_data','cifar') ;
% experiment result directory
opts.expDir = fullfile('cifar_data','cifar-baseline') ;
% image database
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
% set up the batch size (split the data into batches)
opts.train.batchSize = 100 ;
% number of Epoch (iterations)
opts.train.numEpochs = 180;
% resume the train
opts.train.continue = true ;
% use the GPU to train
opts.train.useGpu = true ;
% set the learning rate
opts.train.learningRate = logspace(-3, -5, opts.train.numEpochs); 
% set weight decay
opts.train.weightDecay = 0.0005 ;
% set momentum
opts.train.momentum = 0.9 ;
% experiment result directory
opts.train.expDir = opts.expDir ;
% parse the varargin to opts. 
% If varargin is empty, opts argument will be set as above
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

imdb = load(opts.imdbPath) ;




%% Define network 
% The part you have to modify
net.layers = {} ;

% 1 conv1 (Out: 32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,3,50, 'single'), ...
  'biases', zeros(1, 50, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

% 2 pool1 (max pool)  (Out: 15)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [5 5], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

% 3 conv2 (Out: 9)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(7,7,50,50, 'single'),...
  'biases', zeros(1,50,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 0) ;

% 4 relu2
net.layers{end+1} = struct('type', 'relu') ;
                     
                  
% NEW conv2
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(7,7,50,75, 'single'),...
  'biases', zeros(1,75,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 3) ;

% NEW relu2
net.layers{end+1} = struct('type', 'relu') ;


% NEW conv2
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(7,7,75,75, 'single'),...
  'biases', zeros(1,75,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 3) ;

% NEW relu2
net.layers{end+1} = struct('type', 'relu') ;

% NEW pool2 (avg pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [7 7], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) 
                      
net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;

% 6 conv5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(2,2,75,100, 'single'),...
                           'biases', zeros(1,100,'single'), ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'stride', 1, ...
                           'pad', 0) ;
% 7 loss
net.layers{end+1} = struct('type', 'softmaxloss') ;



% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end
%% display the net
vl_simplenn_display(net);
%% start training
[net,info] = cnn_train_cifar(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 2) , 'test', find(imdb.images.set == 3)) ;
%% Record the result into csv and draw confusion matrix
load(['cifar_data/cifar-baseline/net-epoch-' int2str(opts.train.numEpochs) '.mat']);
load(['cifar_data/cifar-baseline/imdb' '.mat']);
fid = fopen('cifar_prediction.csv', 'w');
strings = {'ID','Label'};
for row = 1:size(strings,1)
    fprintf(fid, repmat('%s,',1,size(strings,2)-1), strings{row,1:end-1});
    fprintf(fid, '%s\n', strings{row,end});
end
fclose(fid);
ID = 1:numel(info.test.prediction_class);
dlmwrite('cifar_prediction.csv',[ID', info.test.prediction_class], '-append');

val_groundtruth = images.labels(45001:end);
val_prediction = info.val.prediction_class;
val_confusionMatrix = confusion_matrix(val_groundtruth , val_prediction);
cmp = jet(50);
figure ;
imshow(ind2rgb(uint8(val_confusionMatrix),cmp));
imwrite(ind2rgb(uint8(val_confusionMatrix),cmp) , 'cifar_confusion_matrix.png');
toc

% --------------------------------------------------------------------
%% call back function get the part of the batch
function [im, labels] = getBatch(imdb, batch , set)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch);

batchSize = size(batch,2);
% data augmentation
if set == 1 % training
    for i = 1 : batchSize
        %mean shift
        %already performed
        
        % fliplr
        if rand() > 0.5
            im(:,:,:,i) = fliplr(im(:,:,:,i));
        end
        % noise
        
        
        % random crop
        if rand() > 0.3
            L = randi([22,31]);
            xInd = randi(32-L+1)+(0:L-1);
            yInd = randi(32-L+1)+(0:L-1);
            origImg = im(:, :, :, i);
            crop = origImg(xInd, yInd, :);
            im(:,:,:,i) = imresize(crop, 32/L);
        end
        % and other data augmentation
    end
end


if set ~= 3
    labels = imdb.images.labels(1,batch) ;
end

