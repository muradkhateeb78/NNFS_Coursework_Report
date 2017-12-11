data = importdata('sortedData.txt');

%Training variables
trainingData_count = 500;
trainParam_goal = 0.095;
trainParam_epochs = 100;
trainParam_max_fail = 20;
inputData = data(:,2:10);
targetData = data(:,11);

%Training Input Data
traintrainInputData2 = inputData(1:trainingData_count/2,:);
traintrainInputData4 = inputData((size(inputData)-trainingData_count/2)+1:size(inputData),:);

%Training Outpu Data
trainOutputData12 = targetData(1:trainingData_count/2,:);
trainOutputData14 = targetData((size(targetData)-trainingData_count/2)+1:size(targetData),:);

%Combining training input data and training output data.
trainInputData = cat(1,traintrainInputData2,traintrainInputData4);
trainOutputData = cat(1,trainOutputData12,trainOutputData14);

%Testing input and output Data
testingInputData = inputData((trainingData_count/2)+1:(size(inputData)-trainingData_count/2),:);
testingOutputData = targetData((trainingData_count/2)+1:(size(targetData)-trainingData_count/2),:);
seed = RandStream('mt19937ar', 'seed', 1);
RandStream.setGlobalStream(seed);

%Neural Network creation
network = newff(trainInputData',trainOutputData',10, {'tansig' 'tansig'}, 'trainr', 'learngd', 'mse');

%Training Parameters
network.trainParam.epochs = trainParam_epochs;
network.trainParam.goal = trainParam_goal;
network.trainParam.max_fail = trainParam_max_fail;

%Training Neural Network
network = train(network,trainInputData',trainOutputData');
resultant = network(testingInputData');
match_counter = 0;

%Normalizing the resultant matrix
resultant = resultant';
for i=1:size(resultant)
    if(resultant(i)<=3)
        resultant(i)=2;
    else
        resultant(i)=4;
    end
    if(testingOutputData(i)==resultant(i))
        match_counter = match_counter+1;
    end
end

%Finding Accuracy
Accuracy = (match_counter/size(resultant,1))*100;