function completed = benchmark_rl_agentDDPG_NavTask()
% benchmark_neuralNetwork_rl_agentDDPG_NavTask - Example for set-based
%   reinforcement learning. This example script trains a point-based,
%   naive-adversarial, grad-adversarial, MAD-adversarial, SA-PC and SA-SC agents with
%   omega = 0 and omega = 0.5 [1]. The results are only evaluated for one
%   random seed. In the example we use a shortened evaluation. To fully
%   train the agents set 'totalEpisodes' to 2000 and initialOps to 'uniform'.
%
% Syntax:
%    res = benchmark_neuralNetwork_rl_agentDDPG_NavTask()
%
% Inputs:
%    -
%
% Outputs:
%    res - boolean
%
% References:
%   [1] Wendl, M. et al. Training Verifiably Robust Agents Using Set-Based
%       Reinforcement Learning, 2024.
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also:

% Authors:       Manuel Wendl
% Written:       27-August-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% System Dynamics ---------------------------------------------------------
f = @(x,u) [x(3)*cos(x(4));x(3)*sin(x(4));u(1);u(2)];
sys = nonlinearSys(f);

% Settings ----------------------------------------------------------------
totalEpisodes = 2000;

% RL settings
options.rl.gamma = .99;
options.rl.tau = .005;
options.rl.expNoise = .2;
options.rl.expNoiseType = 'OU';
options.rl.expDecay = 1;
options.rl.batchsize = 64;
options.rl.buffersize = 1e6;
options.rl.noise = .1;
options.rl.printFreq = 100 ;
options.rl.visRate = 100;

% Actor Settings
options.rl.actor.nn.use_approx_error = true;
options.rl.actor.nn.poly_method = 'bounds';
options.rl.actor.nn.train.optim = nnAdamOptimizer(1e-4,.9,.999,1e-8,0);
options.rl.actor.nn.train.method = 'point';
options.rl.actor.nn.train.eta = .1;
options.rl.actor.nn.train.omega = 0;
options.rl.actor.nn.train.zonotope_weight_update = 'outer_product';
options.rl.actor.nn.train.backprop = true;
options.rl.actor.nn.train.advOps.numSamples = 200;
options.rl.actor.nn.train.advOps.alpha = 4;
options.rl.actor.nn.train.advOps.beta = 4;
options.rl.actor.nn.train.use_gpu = canUseGPU;

% Critic Settings
options.rl.critic.nn.use_approx_error = true;
options.rl.critic.nn.poly_method = 'bounds';
options.rl.critic.nn.train.optim = nnAdamOptimizer(1e-3,.9,.999,1e-8,1e-2);
options.rl.critic.nn.train.method = 'point';
options.rl.critic.nn.train.eta = .01;
options.rl.critic.nn.train.zonotope_weight_update = 'outer_product';
options.rl.critic.nn.train.backprop = true;
options.rl.critic.nn.train.use_gpu = canUseGPU;

% Env settings
options.rl.env.x0 = interval([3;3;0;0],[3;3;0;0]);
options.rl.env.initialOps = 'None';
options.rl.env.dt = .2;
options.rl.env.timeStep = .05;
options.rl.env.maxSteps = 40;
options.rl.env.collisioCheckBool = true;
options.rl.env.evalMode = 'point';
options.rl.env.solver = 'ODE45';
options.rl.env.reach.alg = 'lin';
options.rl.env.reach.zonotopeOrder = 200;

% Allocate Agent Array ----------------------------------------------------
seed = 5;
numAgents = 7;
agents = cell(seed,numAgents);

% Build Environment -------------------------------------------------------

env = ctrlEnvironment(sys,@aux_rewardFun_NavTask,@aux_collisionCheck_NavTask,options);

for i = 1:seed
    % Build NNs ---------------------------------------------------------------

    % Actor NN
    actorLayers = [
        featureInputLayer(4)
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(2)
        tanhLayer
        ];

    % Critic NN
    criticLayers = [
        featureInputLayer(6)
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(1)
        ];

    rng(seed,"twister"); % Set rng
    % initialize weights and biases of the layers
    actorInit = dlnetwork(actorLayers);
    nnActor = neuralNetwork.convertDLToolboxNetwork(num2cell(actorInit.Layers),false);

    % initialize weights and biases of the layers
    criticInit = dlnetwork(criticLayers);
    nnCritic = neuralNetwork.convertDLToolboxNetwork(num2cell(criticInit.Layers),false);

    % Point-Based Actor -------------------------------------------------------
    options.rl.actor.nn.train.method = 'point';
    options.rl.critic.nn.train.method = 'point';

    rng(seed,"twister"); % Set rng
    DDPG1 = agentDDPG(nnActor,nnCritic,options);
    DDPG1 = DDPG1.train(env,totalEpisodes);
    agents{i,1} = DDPG1;

    % Point-Based Actor naive advers attack -------------------------------
    options.rl.actor.nn.train.method = 'naive';
    options.rl.critic.nn.train.method = 'point';

    rng(seed,"twister"); % Set rng
    DDPG2 = agentDDPG(nnActor,nnCritic,options);
    DDPG2 = DDPG2.train(env,totalEpisodes);
    agents{i,2} = DDPG2;

    % Point-Based Actor grad advers attack -------------------------------
    options.rl.actor.nn.train.method = 'grad';
    options.rl.critic.nn.train.method = 'point';

    rng(seed,"twister"); % Set rng
    DDPG3 = agentDDPG(nnActor,nnCritic,options);
    DDPG3 = DDPG3.train(env,totalEpisodes);
    agents{i,3} = DDPG3;

    % Point-Based Actor MAD advers attack -----------------------------------
    options.rl.actor.nn.train.method = 'MAD';
    options.rl.critic.nn.train.method = 'point';
    options.rl.actor.nn.train.advOps.numSamples = 10;

    rng(seed,"twister"); % Set rng
    DDPG4 = agentDDPG(nnActor,nnCritic,options);
    DDPG4 = DDPG4.train(env,totalEpisodes);
    agents{i,4} = DDPG4;

    % Set-Based Actor and Point-Based Critic --------------------------------
    options.rl.actor.nn.train.method = 'set';
    options.rl.critic.nn.train.method = 'point';

    rng(seed,"twister"); % Set rng
    DDPG5 = agentDDPG(nnActor,nnCritic,options);
    DDPG5 = DDPG5.train(env,totalEpisodes);
    agents{i,5} = DDPG5;

    % Set-Based Actor and Set-Based Critic --------------------------------
    options.rl.actor.nn.train.method = 'set';
    options.rl.actor.nn.train.omega = 0;
    options.rl.critic.nn.train.method = 'set';

    rng(seed,"twister"); % Set rng
    DDPG6 = agentDDPG(nnActor,nnCritic,options);
    DDPG6 = DDPG6.train(env,totalEpisodes);
    agents{i,6} = DDPG6;

    % Set-Based Actor and Set-Based Critic --------------------------------
    options.rl.actor.nn.train.method = 'set';
    options.rl.actor.nn.train.omega = .5;
    options.rl.critic.nn.train.method = 'set';

    rng(seed,"twister"); % Set rng
    DDPG7 = agentDDPG(nnActor,nnCritic,options);
    DDPG7 = DDPG7.train(env,totalEpisodes);
    agents{i,7} = DDPG7;

end

cpath = fileparts(mfilename("fullpath"));
scriptName = mfilename;
resultsDir = fullfile(cpath, '../../results', scriptName);

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    cd(resultsDir)
else
    cd(resultsDir)
end

evaluateAgentsStatistically(agents,env,0:0.01:0.1,'inf');

save("Nav_agents.mat","agents")

cd("./../../code/scripts")

completed = true;

end


% Auxiliary functions -----------------------------------------------------

function reward = aux_rewardFun_NavTask(x)

restrictedAreas = {[1,1;2,2]};

if isnumeric(x)
    reward = -sum(abs(x(end,1:2)));
    countViolations = 0;
    for i = 1:length(restrictedAreas)
        violations = x(:,1) >restrictedAreas{i}(1,1) & x(:,1)<restrictedAreas{i}(2,1)&x(:,2)>restrictedAreas{i}(1,2) & x(:,2)<restrictedAreas{i}(2,2);
        countViolations = countViolations + sum(violations);
    end
    reward = reward - countViolations;
else
    w = [1,1,0,0,0,0];
    reward = -max(abs(supportFunc(x.timePoint.set{end},w,'lower')),abs(supportFunc(x.timePoint.set{end},w)));
    
    countViolations = 0;
    for i = 1:length(restrictedAreas)
        I = interval(restrictedAreas{1}(1,:)',restrictedAreas{i}(2,:)');
        for j = 1:length(x.timePoint.set)
            countViolations = countViolations + isIntersecting(zonotope(I),project(x.timePoint.set{j},[1,2]));
        end
    end

    reward = reward -countViolations;
end
end

function collisionBool = aux_collisionCheck_NavTask(x)
if isa(x,'numeric')
    if all(abs(x(:,1:3))<0.1,"all")
        collisionBool = true;
    else
        collisionBool = false;
    end
elseif isa(x,'reachSet')
    if norm(x.timePoint.set{1}.c)<0.1
        collisionBool = true;
    else
        collisionBool = false;
    end
end
end

% ------------------------------ END OF CODE ------------------------------
