% Clear the workspace
clear all; clc;

% Load data
%cd '/Users/Cybelle/Dropbox/5/data'
cd '/Users/dcyoung/Dropbox/5/data'
%cd 'C:/Users/rock-/Dropbox/School/BigData_MachineLearning/AppliedMachineLearning/5/data'
x = load('docword.nips_cropped.txt');

%Load the vocab list
fileID = fopen('vocab.nips.txt');
vocabList = textscan(fileID, '%s');
vocabList = vocabList{1}(:,1);
clear fileID;

% Define parameters
ntopics = 30;   % # of topics/ kind of like # of clusters
ndoc = 1500;    % # of documents/example data items
nwordtypes = 12419;     % # of word types (ie: vocab words)
nwordtokens = 746316;   % # of total instances of all word types
maxiter = 30; %maximum number of iterations allowed for EM algorithm.

% Organize the data by document
xc = { };
for i = 1:ndoc
    xc{i} = x(find(x(:,1)==i),2:3);
end





%%

% Initialize pi vector and p matrix

% Vector of probabilities... 1 for each topic... each value corresponds to
% the probability that the topic was chosen to draw a data item 
pi = rand(ntopics, 1); 
pi = pi./sum(pi);


% Matrix of probabilities that a word is present in a given topic
p = rand(ntopics, nwordtypes); %for even start
for j = 1:ntopics
    p(j,:) = p(j,:)./sum(p(j,:));
end


%Create a vector to quickly access the number of relevant word types per
%document. A word type is relevant if its count is >=1 in the document
numRelevantWordsPerDoc = zeros(1,ndoc);
%create a vector to quickly access the total number of word tokens in a doc
numWordTokensPerDoc = zeros(1,ndoc);
for docID = 1:ndoc
    numRelevantWordsPerDoc(docID) = size(xc{docID},1);
    numWordTokensPerDoc(docID) = sum(xc{docID}(:,2));
end


%%
p_last_iter = p;
pi_last_iter = pi;

for iteration = 1:maxiter
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -----------------   E-step  --------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    display('E-step');
    % temporarily reinitialize all weights to 1, before they are recomputed
    w = repmat(1,ndoc,ntopics);

    % logarithmic form of the E step.
    % see... https://piazza.com/class/ijn48296bq5tc?cid=351
    for i = 1:ndoc
        %calculate Ajs
        logAjVec = repmat(0,1,ntopics);
        for j = 1:ntopics
            logAjVec(j) = log(pi(j));
            for k = 1:numRelevantWordsPerDoc(i)
                wordId = xc{i}(k,1);            
                wordCount = xc{i}(k,2);
                logAjVec(j) = logAjVec(j) + wordCount * log(p(j,wordId));
            end
        end

        % get the max value of any logAj
        [logAmax,ind] = max(logAjVec(:)); 
        % get the index of the max value of any logAj
        %[scrapM,logAmax_index] = ind2sub(size(logAjVec),ind);
        
        % calculate the third term from the final eqn in the above link
        thirdTerm = 0;
        for l = 1:ntopics
            thirdTerm = thirdTerm + exp(logAjVec(l)-logAmax);
        end
        
        % calculate w(i,j)
        for j = 1:ntopics
            logY = logAjVec(j) - logAmax - log(thirdTerm);
            w(i,j) = exp(logY);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -----------------   M-step  --------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    display('M-step')

    %temporarily reinitialize p and pi to 0, before they are recomputed
    p = repmat(0,ntopics,nwordtypes); %matrix of p_j vectors..
    pi = repmat(0,ntopics,1);
        
    %new version following https://piazza.com/class/ijn48296bq5tc?cid=350
    for j = 1:ntopics
        
        % p_j is a vector of p_j,k for all k. 
        % we want to calculate each p_j,k (probability that word k belongs 
        % to topic j) as the ratio of...
        %   1. number of occurrences of word k in all documents clustered into topic j,  to
        %   2. total number of word occurrences in all documents clustered into topic j.
        
        denominatorSum = 0;
        for i = 1:ndoc
            for k = 1:numRelevantWordsPerDoc(i)
                wordId = xc{i}(k,1);            
                wordCount = xc{i}(k,2);
                p(j,wordId) = p(j,wordId) + wordCount*w(i,j);
            end
            denominatorSum =  denominatorSum + numWordTokensPerDoc(i)*w(i,j);
        end
        
        % For each word type k, calculate p_j,k (probability that word k 
        % belongs to topic j). Do this for the whole vector p_j at once
        
        p(j, :) = p(j,:)./denominatorSum; % new way
        % p(j,:) = p(j,:) ./ sum(w(:,j)); % Old way (normalize p)
        
        % Update pi
        pi(j) = sum(w(:,j)) / ndoc;       
    end

    display(pi')
    
    pdiffsq = sum(sum((p - p_last_iter).^2));
    pidiffsq = sum(sum((pi - pi_last_iter).^2));
    
    if (pdiffsq < .000000001 && pidiffsq < .000000001)
        disp('Convergence Criteria Met at Iteration:')
        disp(iteration)
        %break;
    end
    
    p_last_iter = p;
    pi_last_iter = pi;
    
end
% clean up workspace
clear p_last_iter pi_last_iter i j k l denominatorSum docID logAjVec;
clear logAjmax logAmax logY pdiffsq pidiffsq scrapM;
clear thirdTerm iteration ind topic;













%% DISPLAY RESULTS
% produce a graph showing, for each topic, the probability with which the 
% topic is selected.
figure;
topics = 1:30;
bar(topics,pi);
title('Probability topic is selected');
xlabel('Topic'); ylabel('Probability selected'); 
xlim([0 31]);
clear topics;

% Produce a table showing, for each topic, the 10 words with the highest 
% probability for that topic.
nDesiredWords = 10;
highProbWordIDs = repmat(0, ntopics, nDesiredWords);
highProbWords = {}; 
for j = 1:ntopics
    [ b, ix ] = sort( p(j,:), 'descend' );
    % First get the indices of the 10 most probable words from each topic
    highProbWordIDs(j,:) = ix(1:nDesiredWords);
    %then grab the actual words from the vocab list
    for k=1:nDesiredWords
        idInVocabList = highProbWordIDs(j,k);
        highProbWords{j,k} = vocabList(idInVocabList);
    end
end
clear j k fileID wordForTopicJ topic b ix; %clean up workspace;





