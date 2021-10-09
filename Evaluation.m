clear all;
close all;
trainPath='E:\Year 4\Computer Vision\cw1\CWMaterial\FaceDatabase\Train\'; % provide full path here
testPath='E:\Year 4\Computer Vision\cw1\CWMaterial\FaceDatabase\Test\';
%% Baseline Method
tic;
   outputLabel=FaceRecognition(trainPath, testPath);
baseLineTime=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
    if strcmp(outputLabel(i,:),testLabel(i,:))
        correctP=correctP+1;
    end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%

%% Method 1 developed by you
tic;
  outputLabel1=FaceRecognition1(trainPath, testPath);
method1Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
   if strcmp(outputLabel1(i,:),testLabel(i,:))
       correctP=correctP+1;
   end
end

recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%


%% Method 2 developed by you
tic;
  outputLabel2=FaceRecognition2(trainPath, testPath);
method2Time=toc

load testLabel
correctP=0;
for i=1:size(testLabel,1)
   if strcmp(outputLabel2(i,:),testLabel(i,:))
       correctP=correctP+1;
   end
end
recAccuracy=correctP/size(testLabel,1)*100  %Recognition accuracy%
