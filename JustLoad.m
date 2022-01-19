function data=JustLoad(data)

% data=load('New6');
% data=data.data;
Inputs=data.Inputs';
Targets=data.Targets';
Targets=Targets(:,1);
nSample=size(Inputs,1);
    
% Shuffle Data
% S=randperm(nSample);
% Inputs=Inputs(S,:);
% Targets=Targets(S,:);

% Train Data
pTrain=1.0;
nTrain=round(pTrain*nSample);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain,:);
% Test Data
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end,:);
% Export
data.TrainInputs=TrainInputs;
data.TrainTargets=TrainTargets;
data.TestInputs=TestInputs;
data.TestTargets=TestTargets;
end