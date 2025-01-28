function main(eval_name)
disp('Startup ...')

% Check if everything is set up correctly:
check = test_requiredToolboxes;
if check
    disp('All Required Toolboxes are installed.')
else
    disp("Make sure all required toolboxes are installed!")
    disp("See Manual: https://tumcps.github.io/CORA")
    return
end


% Possible running modes
% - 'dummy': default, executes a demo training with shorter runtime than evaluation scripts
% - 'eval': executes all evaluations scripts

mode = 'dummy';

if strcmp(mode,'dummy')
	% Run a demo training for the Quadrocopter 1D Benchmark
	dummy_benchmark_rl_agentDDPG_Quad1D
elseif strcmp(mode,'eval')
	% Run full evaluation 
	benchmark_rl_agentDDPG_Quad1D
	benchmark_rl_agentTD3_Quad1D
	benchmark_rl_agentDDPG_Pendulum
	benchmark_rl_agentDDPG_NavTask
	benchmark_rl_agentDDPG_Quad2D
end
end
