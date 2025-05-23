function [obj, loss] = train(obj,critic,batch,options,noiseBatchG)
% train - actor neural network 
%   The actor is trained with randomly sampled batches from the experience
%   buffer. 
%
% Syntax:
%   [obj, loss] = train(obj,critic,batch,options,noiseBatchG)
%
% Inputs:
%   critic - critic with current parameters
%   batch - random batch from replay buffer
%   options.rl - options for RL:
%        .actor.nn - Evaluation paramteres for the actor network
%           .train - Training parameters for the actor network:
%               .use_gpu: true if available (default) Use CPU for training.
%               .optim: nnAdamOptimizer(1e-3,.9,.999,1e-8,1e-2) (default)
%                       Actor optimizer.
%               .backprop: true (default) Training boolean for actor.
%               .method: 'point'(default) Training method for actor:
%                   'point' Standard point-based training 
%                   'set' Set-based training [1]
%                   'random' Random adv. samples form perturbation ball
%                   'extreme' Adv. samples from edges of perturbation ball
%                   'naive' Adv. samples from naive algorithm [2]
%                   'grad' Adv. samples from grad algorthm [2]
%		    'MAD' Adv. training with maximum-action-difference loss [3]
%               .eta: 0.01 (default) Weighting factor for set-based 
%                   training of the actor.
%               .zonotope_weight_update: 'outer_product' (default)
%                   Zonontope weight update for learnable params \theta
%   noiseBatchG - Pre-allocated noise batch
%
% Outputs:
%   obj - updated actor
%   loss - actor loss 
% 
% Refernces:
%   [1] Wendl, M. et al. Training Verifiably Robust Agents Using Set-Based 
%       Reinforcement Learning, 2024
%   [2] Pattanaik, A. et al. Robust Deep Reinforcement Learning with 
%       Adversarial Attacks, Int. Conf. on Autonomous Agents and Multiagent 
%       Systems (AAMAS) 2018
%   [3] H. Zhang et.al. Robust Deep Reinforcement Learning against Adversarial 
%	Perturbations on State Observations, Int. Conf. on Neural Information 
%	Processing Systems (NeurIPS) 2020 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: actor

% Authors:       Manuel Wendl
% Written:       03-November-2023
% Last update:   19-September-2024 (TL, renamed to train)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------
    
cBatch = batch{1};
GBatch = noiseBatchG;

if ~strcmp(options.rl.actor.nn.train.method,'set')
    % Point based learning
    actionBatchC = obj.nn.evaluate_(cBatch,options.rl.actor,obj.idxLayer);
    [loss,policyGradient] = critic.getPolicyGradient(batch,actionBatchC,[],options);
    obj.nn.backprop(policyGradient.gradC,options.rl.actor,obj.idxLayer);
    
    if strcmp(options.rl.actor.nn.train.method,'MAD')
    gradsW = cell(length(obj.idxLayer),1);
    gradsb = cell(length(obj.idxLayer),1);
    for l = obj.idxLayer
        if isa(obj.nn.layers{l},"nnLinearLayer")
            gradsW{l}=obj.nn.layers{l}.backprop.grad.W;
            gradsb{l}=obj.nn.layers{l}.backprop.grad.b;
        end
    end
    advBatch = aux_computeAdvBatch(obj,cBatch,options);
    advActionBatchC = obj.nn.evaluate_(advBatch,options.rl.actor,obj.idxLayer);
    [loss2,madGradient] = aux_computeMADLoss(advActionBatchC,actionBatchC);
    obj.nn.backprop(madGradient,options.rl.actor,obj.idxLayer);
    
    for l = obj.idxLayer
        if isa(obj.nn.layers{l},"nnLinearLayer")
            obj.nn.layers{l}.backprop.grad.W = obj.nn.layers{l}.backprop.grad.W + gradsW{l};
            obj.nn.layers{l}.backprop.grad.b = obj.nn.layers{l}.backprop.grad.b + gradsb{l};
        end
    end
    loss.center = loss.center + loss2;
    end
    loss.vol = 0;
else
    % Set based Learinng
    [actionBatchC,actionBatchG] = obj.nn.evaluateZonotopeBatch_(cBatch,GBatch,options.rl.actor,obj.idxLayer);
   
    [loss,policyGradient] = critic.getPolicyGradient(batch,actionBatchC,actionBatchG,options,noiseBatchG);

    if strcmp(options.rl.critic.nn.train.method,'point') 
        [loss.vol,gradOutG] = aux_computeVolumeLoss(actionBatchG);
    elseif strcmp(options.rl.critic.nn.train.method,'set')
        [loss.vol,gradOutG] = aux_computeVolumeLoss(actionBatchG);
        gradOutG = options.rl.actor.nn.train.omega * gradOutG + (1-options.rl.actor.nn.train.omega) * policyGradient.gradG;
    else
        throw(CORAerror("CORA:notDefined",'Other trainig methods than point or set are not implemented for the critic.'))
    end

    % Scale volume gradients
    gradOutG = options.rl.actor.nn.train.eta/max(options.rl.noise,[],'all') * gradOutG;
    loss.vol =  1/max(options.rl.noise,[],'all')*loss.vol;

    obj.nn.backpropZonotopeBatch_(policyGradient.gradC,gradOutG,options.rl.actor,obj.idxLayer);

end

obj.optim = obj.optim.step(obj.nn,options.rl.actor,obj.idxLayer);

end


% Auxiliary functions -----------------------------------------------------

function [loss,gradOutG] = aux_computeVolumeLoss(yPredG)
loss = 1/size(yPredG,3)*sum(log(2*sum(abs(yPredG),2)),'all');
gradOutG = 1/size(yPredG,3)*1/(sum(abs(yPredG),2)).*sign(yPredG);
nanIdx = isnan(gradOutG)|isinf(gradOutG);
gradOutG(nanIdx) = 0;
end

function [loss,grad] = aux_computeMADLoss(aAdv,aTrue)
loss = 1/size(aAdv,2)*sum((aAdv-aTrue).^2,'all');
grad = aAdv - aTrue;
end

function adv_batch = aux_computeAdvBatch(obj,cBatch,options)
    options.rl.actor.nn.train.updateGrad = false;
    n = options.rl.actor.nn.train.advOps.numSamples;
    alpha = options.rl.actor.nn.train.advOps.alpha;
    
    epsilon = options.rl.noise;
    
    orig_action = obj.nn.evaluate_(cBatch,options.rl.actor,obj.idxLayer);

    adv_state = cBatch + randn(size(cBatch))*1e-5;

    for i = 1:n
        loss_grad = obj.nn.evaluate_(adv_state,options.rl.actor,obj.idxLayer) - orig_action; 
        grad = obj.nn.backprop(loss_grad,options.rl.actor,obj.idxLayer);
        noise_factor = 1e-5/(i+2);
        update_step = grad + noise_factor.* randn(size(adv_state)) * alpha;
        adv_state = adv_state + update_step;
        adv_state = max(min(adv_state,cBatch+epsilon),cBatch-epsilon);
    end
    
    adv_batch = adv_state;

    options.rl.actor.nn.train.updateGrad = true;
end

% ------------------------------ END OF CODE ------------------------------
