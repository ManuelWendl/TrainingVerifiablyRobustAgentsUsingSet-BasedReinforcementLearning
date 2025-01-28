classdef buffer
% buffer - replay buffer of the rl agent.
%   the array contains (s_i,a_i,r_i,s_i+1) and visualData for
%   plotting of trajectories during training process
%
% Syntax:
%   obj = buffer(bufferSize,use_gpu)
%
% Inputs:
%   bufferSize - maximal size of buffer
%   use_gpu - boolean
% 
% Outputs:
%   obj - instantiated buffer object
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: DDPGagent

% Authors:       Manuel Wendl
% Written:       24-October-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

    properties
        array
        visualisationData
        bufferSize
        currentIndx
        use_gpu
    end
    
    methods
        % constructor 
        function obj = buffer(bufferSize,use_gpu)
            obj.use_gpu = use_gpu;
            obj.array = {};
            obj.visualisationData.episodeNum = [];
            obj.visualisationData.reachSet = {};
            obj.bufferSize = bufferSize;
            obj.currentIndx = 1;
        end

        function obj = resetBuffer(obj)
            obj.array = {[],[],[],[],[]};
            obj.visualisationData.episodeNum = [];
            obj.visualisationData.reachSet = {};
            obj.currentIndx = 1;
        end
    end
end

% ------------------------------ END OF CODE ------------------------------
