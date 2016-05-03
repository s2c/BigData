%% Define network 
% The part you have to modify
net.layers = {} ;




%L1: conv1 
%(input = 32x32x3, Ouput = 32x32x64)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,3,64, 'single'), ...
  'biases', zeros(1, 64, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

%L5 relu1
%(input = 32x32x64, Ouput = 32x32x64)
net.layers{end+1} = struct('type', 'relu') ;

%L3: pool1 (max pool)
%(input = 32x32x64, Ouput = 16x16x64)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

%L4: norm1 (local response normalization)
%(input = 16x16x64, Ouput = 16x16x64)
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [4 1 0.001/9 0.75]) ;


%L5: conv2 
%(input = 16x16x64, Ouput = 16x16x64)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,64,64, 'single'), ...
  'biases', zeros(1, 64, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

%L6 relu2
%(input = 16x16x64, Ouput = 16x16x64)
net.layers{end+1} = struct('type', 'relu') ;

%L6: norm2 (local response normalization)
%(input = 16x16x64, Ouput = 16x16x64)
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [4 1 0.001/9 0.75]) ;

%L7: pool2 (max pool)
%(input = 16x16x64, Ouput = 8x8x64)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

%L8: conv3
%(input = 8x8x64 Ouput = 8x8x64)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(3,3,64,64, 'single'), ...
  'biases', zeros(1, 64, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 1) ;

%L9 relu3
%(input = 8x8x64, Ouput = 8x8x64)
net.layers{end+1} = struct('type', 'relu') ;

%L10: pool3 (max pool)
%(input = 8x8x64, Ouput = 2x2x64)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [7 7], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;


%L11: conv4
%(input = 2x2x64 Ouput = 1x1x100)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(2,2,64,100, 'single'), ...
  'biases', zeros(1, 100, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 0) ;

%L12 loss 
%(input = 1x1x100, Ouput = 1x100)
net.layers{end+1} = struct('type', 'softmaxloss') ;
