function  outputLabel=FaceRecognition1(trainPath, testPath)
%%   A simple face reconition method using cross-correlation based tmplate matching.
%    trainPath - directory that contains the given training face images
%    testPath  - directory that constains the test face images
%    outputLabel - predicted face label for all tested images 

%% Retrieve training images and labels
folderNames=ls(trainPath);
trainImgSet=zeros(600,600,3,length(folderNames)-2); % all images are 3 channels with size of 600x600
labelImgSet=folderNames(3:end,:); % the folder names are the labels
for i=3:length(folderNames)
    imgName=ls([trainPath, folderNames(i,:),'\*.jpg']);
    trainImgSet(:,:,:,i-2)= imread([trainPath, folderNames(i,:), '\', imgName]);
end

%% Prepare the training images
% Initialize the hog extraction cell size, image size.
cellSize = [8, 8];
imageSize = [128, 128];

% All images use 8100 feature vector.
numOfFeatureVector = 8100;
trainHogFeatureSets=zeros(size(trainImgSet,4), numOfFeatureVector);

faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');% Creates a face detector.

% Loop through all training images.
for i=1:size(trainImgSet,4)
      tmpImg = imresize(uint8(trainImgSet(:, :, :, i)), imageSize); % Resize training image according to image size.
      faceImg = detectFace(faceDetector, tmpImg, imageSize); % Extract face region using face detector
      faceImg = rgb2gray(faceImg); % Transfer img from RGB to Gray.
      faceImg = histeq(faceImg); % Enhance contrast using histogram qqualization
      [trainHogFeature, ~] = extractHOGFeatures(faceImg, 'CellSize', cellSize); % Extract HOG feature with cell size.
      trainHogFeatureSets(i, :) = trainHogFeature; % Store HOG feature
end

%% Train a multi-class SVM classifier .
classifer = fitcecoc(trainHogFeatureSets,labelImgSet, 'Coding', 'onevsall');   % Use fitcecoc to train a SVM classifier with 'onevsall' encoding scheme


%% Prepare the testing images
testImgNames=ls([testPath,'*.jpg']);
testHogFeatureSets = zeros(size(testImgNames, 1), numOfFeatureVector);
for i=1:size(testImgNames,1)
      testImg=imread([testPath, testImgNames(i,:)]);
      testImg = imresize(uint8(testImg), imageSize); % Resize testing image according to image size.
      testFaceImg = detectFace(faceDetector, testImg, imageSize); % Extract face region using face detector
      testFaceImg = rgb2gray(testFaceImg);  % Transfer img from RGB to Gray.
      testFaceImg = histeq(testFaceImg); % Enhance Contrast using histogram equalization
      [testHogFeature, ~] = extractHOGFeatures(testFaceImg, 'CellSize', cellSize); % Extract HOG feature with cell size.
      testHogFeatureSets(i, :) = testHogFeature; % Store HOG feature
end

%% Use trained SVM to predict results
outputLabel = predict(classifer, testHogFeatureSets);
end

% Detect faces and returned the face image with specific image size using the Viola-Jones algorithm.
function outputFaceImg = detectFace(detector, img, imgSize) 
   
    faceBoundingBox = detector(img); % Extract the bounding box using detector.
    if ~isempty(faceBoundingBox)
        if (size(faceBoundingBox,1) == 1) 
            img = imcrop(img, faceBoundingBox); % Crop img according to the predicted bounding box.
        else % If more than one bounding box are predicted
            [~, row] = max(faceBoundingBox(:, 3) .* faceBoundingBox(:, 4)); % Select the bounding box has the largest region
            img = imcrop(img, faceBoundingBox(row, :)); % Crop img according to the selected bounding box.
        end
    end
    % Resize the detected face img to a specific size.
    outputFaceImg = imresize(img, imgSize);
end

