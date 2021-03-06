%% Define network 
% The part you have to modify
net.layers = {} ;

% 1 conv1
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 1e-4*randn(5,5,3,50, 'single'), ...
  'biases', zeros(1, 50, 'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 2) ;


% 2 pool1 (max pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [5 5], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;

% 3 conv2
net.layers{end+1} = struct('type', 'conv', ...
  'filters', 0.01*randn(7,7,50,60, 'single'),...
  'biases', zeros(1,60,'single'), ...
  'filtersLearningRate', 1, ...
  'biasesLearningRate', 2, ...
  'stride', 1, ...
  'pad', 0) ;

% 4 relu2
net.layers{end+1} = struct('type', 'relu') ;

% 5 pool2 (avg pool)
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [7 7], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; 
                      
% 6 conv5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.1*randn(2,2,60,100, 'single'),...
                           'biases', zeros(1,100,'single'), ...
                           'filtersLearningRate', 1, ...
                           'biasesLearningRate', 2, ...
                           'stride', 1, ...
                           'pad', 0) ;
% 7 loss
net.layers{end+1} = struct('type', 'softmaxloss') ;