
% Clear the workspace
clear all; clc;

% Set the workspace
%cd '/Users/Cybelle/Dropbox/5/images'
cd '/Users/dcyoung/Dropbox/5/images'
%cd 'C:/Users/rock-/Dropbox/School/BigData_MachineLearning/AppliedMachineLearning/5/images'

% Potential image names
imgName = 'polarlights';


% Load the image
x = imread([imgName, '.jpg']);

%Define Parameters
nSegments = 20;   % # of color clusters
nPixels = size(x,1)*size(x,2);    % # of pixels
maxIterations = 30; %maximum number of iterations allowed for EM algorithm.
nColors = 3;

%Determine the output path for writing images to files
outputPath = ['output/', num2str(nSegments), '_segments/', imgName, '_randStart/'];
imwrite(x, [outputPath, num2str(0), '.png']);


%reshape the image into a single vector of pixels for easier loops
pixels = reshape(x,nPixels,nColors,1);
pixels = double(pixels);


% Initialize pi vector and p matrix

% Vector of probabilities for segments... 1 value for each segment.
% Best to think of it like this...
% When the image was generated, color was determined for each pixel by selecting
% a value from one of "n" normal distributions. Each value in this vector 
% corresponds to the probability that a given normal distribution was chosen.
pi = rand(nSegments, 1); %repmat(1/nSegments, nSegments, 1); 
pi = pi./sum(pi);
%add noise the initialization (but keep it unit)
%for j = 1:length(pi)
%    if(mod(j,2)==1)
%        increment = normrnd(0,.0001);
%        pi(j) = pi(j) + increment;
%    else
%        pi(j) = pi(j) - increment;
%    end
%end



% Matrix of mean color (RGB vec)for the "n" distributions that will end up yield "n" image segments
mu = rand(nSegments,nColors);%repmat(1/nSegments,nSegments,nColors); %for even start
for j = 1:nSegments
    mu(j,:) = mu(j,:)./sum(mu(j,:));
end
%add noise to the initialization (but keep it unit)
%for j = 1:nSegments
%    if(mod(j,2)==1)
%        increment = normrnd(0,.0001);
%    end
%    for k = 1:nColors
%         if(mod(j,2)==1)
%            mu(j,k) = mean(pixels(:,k)) + increment;
%         else
%             mu(j,k) = mean(pixels(:,k)) - increment;
%         end
%         if(mu(j,k) < 0)
%             mu(j,k) = 0;
%         end
%    end
%end


mu_last_iter = mu;
pi_last_iter = pi;

for iteration = 1:maxIterations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -----------------   E-step  --------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    display('E-step');


    % Weights that describe the likelihood that pixel "i" belongs to a color cluster "j"
    w = repmat(1,nPixels,nSegments);  % temporarily reinitialize all weights to 1, before they are recomputed

    % logarithmic form of the E step.
    % see... https://piazza.com/class/ijn48296bq5tc?cid=351 but note that 
    % new logarithmic version was derived for the non-topic based model
    for i = 1:nPixels
        % Calculate Ajs
        logAjVec = repmat(0,1,nSegments);
        for j = 1:nSegments
            logAjVec(j) = log(pi(j)) - .5*((pixels(i,:)-mu(j,:))*((pixels(i,:)-mu(j,:)))');
        end

        % Note the max
        [logAmax,ind] = max(logAjVec(:)); 

        % Calculate the third term from the final eqn in the above link
        thirdTerm = 0;
        for l = 1:nSegments
            thirdTerm = thirdTerm + exp(logAjVec(l)-logAmax);
        end

        % Calculate w(i,j)
        for j = 1:nSegments
            logY = logAjVec(j) - logAmax - log(thirdTerm);
            w(i,j) = exp(logY);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % -----------------   M-step  --------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    display('M-step')

    %temporarily reinitialize mu and pi to 0, before they are recomputed
    mu = repmat(0,nSegments,nColors); % mean color for each segment
    pi = repmat(0,nSegments,1);

    %new version following https://piazza.com/class/ijn48296bq5tc?cid=350
    for j = 1:nSegments

        denominatorSum = 0;
        for i = 1:nPixels
            mu(j,:) = mu(j,:) + pixels(i,:).*w(i,j);
            denominatorSum = denominatorSum + w(i,j);
        end

        % Update mu
        mu(j,:) =  mu(j,:) ./ denominatorSum;

        % Update pi
        pi(j) = sum(w(:,j)) / nPixels;       
    end

    display(pi')

    muDiffSq = sum(sum((mu - mu_last_iter).^2));
    piDiffSq = sum(sum((pi - pi_last_iter).^2));

    if (muDiffSq < .0000001 && piDiffSq < .0000001)
        disp('Convergence Criteria Met at Iteration:')
        disp(iteration)
        break;
    end

    mu_last_iter = mu;
    pi_last_iter = pi;


    % Draw the segmented image using the mean of the color cluster as the 
    % RGB value for all pixels in that cluster.
    segpixels = pixels;
    cluster = 0;
    for i = 1:nPixels
        cluster = find(w(i,:) == max(w(i,:)));
        segpixels(i,:) = (mu(cluster,:));
    end

    segpixels = reshape(segpixels,size(x,1),size(x,2),nColors);
    segpixels = segpixels ./255; %normalize each entry to make img of type double
    imwrite(segpixels, [outputPath, num2str(iteration), '.png']);

end
% clean up workspace
clear mu_last_iter pi_last_iter i j k l denominatorSum logAjVec;
clear logAjmax logAmax logY muDiffSq piDiffSq;
clear thirdTerm;












