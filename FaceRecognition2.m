function  outputLabel=FaceRecognition2(trainPath, testPath)
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

%% Preprocess the training image by extracting the face region
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); % Initialize a face detector
imageSize = [160, 160]; % Initialize the defined image size.
trainFaceSet=zeros(160, 160, 3, length(folderNames)-2); 

for i = 1:size(trainImgSet, 4)
    tmpImg = imresize(uint8(trainImgSet(:, :, :, i)), imageSize); % Resize the training image according to image size
    faceImg = detectFace(faceDetector, tmpImg, imageSize); % Extract face region using face detector
    % Standardize the face image
    imgMean = mean2(faceImg);
    imgStd = std2(faceImg);
    standardizeImg = (double(faceImg) - imgMean) / imgStd;
    trainFaceSet(:, :, :, i) = standardizeImg;
end

%% Network Initialization
facenetKeras = importKerasLayers('facenet_keras.h5', 'ImportWeights', true); % Import Keras Model
placeholderLayers = findPlaceholderLayers(facenetKeras); % Replace the layers that cannot be recognized by MATLAB

% Replace each invalid ScaleSum Layer with one scaling layer and one
% addition layer
for i = 1:size(placeholderLayers, 1)
    additionLayerName = ['addition_', int2str(i)];
    customAdditionLayer = additionLayer(2, 'Name', additionLayerName); % Create a custom addition layer with 2 inputs
    facenetKeras = replaceLayer(facenetKeras, placeholderLayers(i, 1).Name, customAdditionLayer);
    scalingLayerName = ['scaling_', int2str(i)];
    % Create a scaling level with specific scale parameter
    if i <= 5
        blockName = ['Block35_', int2str(i), '_Conv2d_1x1'];
        customScalingLayer = scalingLayer('Scale', 0.17, 'Name', scalingLayerName);
    elseif i >= 6 && i <=15
        blockName = ['Block17_', int2str(i-5), '_Conv2d_1x1'];
        customScalingLayer = scalingLayer('Scale', 0.1, 'Name', scalingLayerName);
    else
        blockName = ['Block8_', int2str(i-15), '_Conv2d_1x1'];
        customScalingLayer = scalingLayer('Scale', 0.2, 'Name', scalingLayerName);
    end
    facenetKeras = disconnectLayers(facenetKeras, blockName, [additionLayerName, '/in2']); % Disconnet the right branch of custom addition layer with the block.
    facenetKeras = addLayers(facenetKeras, customScalingLayer); % Add scaling layer to the network.
    facenetKeras = connectLayers(facenetKeras, blockName, scalingLayerName); % Connect scaling layer with the block.
    facenetKeras = connectLayers(facenetKeras, scalingLayerName, [additionLayerName, '/in2']); % Connect addition layer after the scaling layer.
end
facenetKeras = addLayers(facenetKeras,regressionLayer('Name','output')); % Add a fake output layer to assemble the whole network only.
facenetKeras = connectLayers(facenetKeras, 'Bottleneck_BatchNorm', 'output'); % Connect 128-embedding output layer with previously created output layer. 
faceNet = assembleNetwork(facenetKeras); % Regenerate the complete network structure.
embeddingLayer = 'Bottleneck_BatchNorm'; 

%% Use FaceNet to extract the feature embedding for training image
trainFeatureSet = activations(faceNet, trainFaceSet, embeddingLayer, 'OutputAs', 'rows'); % Get the layer output 

%% Face recognition for the test images
testImgNames=ls([testPath,'*.jpg']);
outputLabel=[];
for i = 1:size(testImgNames, 1)
    testImg=imread([testPath, testImgNames(i,:)]); 
    testImg = imresize(uint8(testImg), imageSize); % Resize testing image according to image size
    testFaceImg = detectFace(faceDetector, testImg, imageSize); % Extract face region using face detector
    % Standardize the face image
    testImgMean = mean2(testFaceImg);
    testImgStd = std2(testFaceImg);
    standardizeTestImg = (double(testFaceImg) - testImgMean) / testImgStd;
    
    featureTest = activations(faceNet, standardizeTestImg, embeddingLayer, 'OutputAs', 'rows'); % Get the face image embedding
    
    % Calculate the euclidean distance between current test image embedding
    % with all embeddings in training feature set. 
    for j = 1:size(trainImgSet, 4)
        euclideanDistance(j) = sqrt(sum((featureTest - trainFeatureSet(j, :)) .^ 2));
    end
    labelIndx = find(euclideanDistance==min(euclideanDistance)); % Retrieve the label that corresponding to the smallest distance.
    outputLabel=[outputLabel;labelImgSet(labelIndx(1),:)];% Store the outputLabels for each of the test image
end
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


