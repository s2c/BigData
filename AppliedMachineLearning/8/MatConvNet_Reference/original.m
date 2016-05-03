
%% Define network 
% The part you have to modify
net.layers = {} ;

% 1 conv1
%(input = 32x32x3, Ouput = 32x32x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,3,32, 'single'), ...
  'biases', zeros(1, 32, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;

% 2 pool1 (max pool)
%(input = 32x32x32, Ouput = 15x15x32)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [5 5], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

% 3 conv2
%(input = 15x15x32, Ouput = 9x9x32)
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(7,7,32,32, 'single'),...
  'biases', zeros(1,32,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 0) ;

% 4 relu2
%(input = 9x9x32, Ouput = 9x9x32)
net.layers{end+1} = struct('type', 'relu') ;

% 5 pool2 (avg pool) 
%(input = 9x9x32, Ouput = 2x2x100)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [7 7], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; 
                      
% 6 conv5 
%(input = 2x2x32, Ouput = 1x1x100)
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(2,2,32,100, 'single'),...
                           'biases', zeros(1,100,'single'), ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'stride', 1, ...
                           'pad', 0) ;
% 7 loss 
%(input = 1x1x100, Ouput = 1x100)
net.layers{end+1} = struct('type', 'softmaxloss') ;
