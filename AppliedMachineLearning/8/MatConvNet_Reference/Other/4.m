
%% Define network 
% The part you have to modify
net.layers = {} ;

% 1 conv1
%(input = 32x32x3, Ouput = 32x32x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(3,3,3,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 1) ;

% 2 relu1
%(input = 32x32x32, Ouput = 32x32x32)
net.layers{end+1} = struct('type', 'relu') ;

% 3 conv2
%(input = 32x32x32, Ouput = 32x32x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(3,3,32,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 1) ;

% 4 relu2
%(input = 32x32x32, Ouput = 32x32x32)
net.layers{end+1} = struct('type', 'relu') ;


% 5 pool1 (max pool)
%(input = 32x32x32, Ouput = 16x16x32)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 0 0 0]) ;

% 6 conv3
%(input = 16x16x32, Ouput = 16x16x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(3,3,32,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 1) ;

% 7 relu3
%(input = 16x16x32, Ouput = 16x16x32)
net.layers{end+1} = struct('type', 'relu') ;


% 8 pool2 (max pool)
%(input = 16x16x32, Ouput = 8x8x32)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [0 0 0 0]) ;
                      
% 9 conv4
%(input = 8x8x32, Ouput = 8x8x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(3,3,32,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 1) ;

% 7 relu4
%(input = 8x8x32, Ouput = 8x8x32)
net.layers{end+1} = struct('type', 'relu') ;

% 8 pool3 (max pool)
%(input = 8x8x32, Ouput = 2x2x32)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [6 6], ...
                           'stride', 2, ...
                           'pad', [0 0 0 0]) ;                       

% 9 conv5 
%(input = 2x2x32, Ouput = 1x1x100)
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(2,2,32,100, 'single'),...
                           'biases', zeros(1,100,'single'), ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'stride', 1, ...
                           'pad', 0) ;
% 10 loss 
%(input = 1x1x100, Ouput = 1x100)
net.layers{end+1} = struct('type', 'softmaxloss') ;
