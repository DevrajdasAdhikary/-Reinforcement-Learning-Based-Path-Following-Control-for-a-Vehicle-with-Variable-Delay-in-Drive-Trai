model = "environment";
open(model);
doReset = true;    % if you want to freshly create model,agents
doTraining = true;  % if you want to newly train the agents
doSimulation = false; % if you want to simulate after training
%% training/simulation length
simlen=80;
maxepisodes  = 3000;
Ts = 0.5;
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.98, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
agentOpts.SaveExperienceBufferWithAgent = 0;

%% Agent observation/action dimension
ObsDim = 8;
ActDim = 2;
obsInfo = rlNumericSpec([ObsDim 1]);
obsInfo.Name = 'observations';  
numObservations = obsInfo.Dimension(1);
actInfo = rlNumericSpec([ActDim 1]);
actInfo.Name = 'actions';
numActions = actInfo.Dimension(1);
%% critic network
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','StateCon')
    fullyConnectedLayer(500,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(350,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','ActionCon')
    fullyConnectedLayer(350,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'StateCon'},'Action',{'ActionCon'},criticOpts);

ulim = 100;
%% actor network
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','StateCon')
    fullyConnectedLayer(500, 'Name','actorFC1')
    reluLayer('Name','actorRelu1')
    fullyConnectedLayer(400, 'Name','actorFC2')
    reluLayer('Name','actorRelu2')
    fullyConnectedLayer(numActions,'Name','actorFC3')
    tanhLayer('Name','actorTanhCon')
    scalingLayer('Name','actorScalingCon','Scale',ulim)];
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,...
    actInfo,'Observation',{'StateCon'},'Action',{'actorScalingCon'},...
    actorOptions);

%% environment creation
agents = [];
observations = {};
actions = {};
agentObjs=[];
target = [];
targetAvgReward = 10000;
agentp = "/ddpg";

ddpgagent = rlDDPGAgent(actor, critic);

agents = [agents model+agentp];
observations{end+1} = obsInfo;
actions{end+1} = actInfo;
agentObjs = [agentObjs ddpgagent];
target = [target targetAvgReward];

env = rlSimulinkEnv(model, agents, obsInfo, actInfo);

%% training
maxsteps = simlen;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',10, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',target,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',target);
if doTraining
    agentOpts.ResetExperienceBufferBeforeTraining = false;
    trainingStats = train(agentObjs, env, trainOpts);
end

if doSimulation
    simOpts = rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
    experiences = sim(env,agentObjs,simOpts);
end
