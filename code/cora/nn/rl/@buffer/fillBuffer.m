function obj = fillBuffer(obj,data,options)
% fillBuffer - fill replay buffer with data
%
% Syntax:
%   obj = fillBuffer(obj,data,options)
%
% Inputs:
%   data - transition (s_i,a_i,r_i,s_i+1,flag)
%   options.rl - reinforcment learning options
% 
% Outputs:
%   obj - updated buffer
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: buffer

% Authors:       Manuel Wendl
% Written:       03-November-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

inputArgsCheck({ ...
    {data, 'att', 'cell'}
    })

if any(size(data) ~= [1,5])
    throw(CORAerror("CORA:specialError","Input data has to contain state, action, reward, next state and termination flag"));
end

obj.array{1}(:,obj.currentIndx) = data{1};

if strcmp(options.rl.critic.nn.train.method,'set')
     obj.array{2}(:,:,obj.currentIndx) = cat(2,data{2}.c,data{2}.G,zeros(size(data{2}.c,1),options.rl.critic.nn.train.num_init_gens-size(data{2}.G,2),'like',single(1)));
else
     obj.array{2}(:,obj.currentIndx) = data{2};
end

obj.array{3}(:,obj.currentIndx) = data{3};
obj.array{4}(:,obj.currentIndx) = single(data{4});
obj.array{5}(:,obj.currentIndx) = single(data{5});

if obj.currentIndx == obj.bufferSize
    obj.currentIndx = 1;
else
    obj.currentIndx = obj.currentIndx+1;
end
end

% ------------------------------ END OF CODE ------------------------------
