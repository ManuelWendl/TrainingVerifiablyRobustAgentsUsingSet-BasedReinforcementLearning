function res = test_spaceex2cora_hybrid_flat_07
% test_spaceex2cora_hybrid_flat_07 - test for model conversion from SpaceEx
%    to CORA for a simple hybrid system with four locations
%
% Syntax:
%    test_spaceex2cora_hybrid_flat_07
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false

% Authors:       Mark Wetzlinger
% Written:       11-January-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------
 
% assume true
res = true;

% directory to SpaceEx model file
dir_spaceex = [CORAROOT filesep 'unitTests' filesep 'converter' ...
    filesep 'spaceex2cora' filesep 'testSystems'];

% file name of SpaceEx model file
filename = 'test_hybrid_flat_fourloc';

% convert SpaceEx model from .xml file
spaceex2cora([dir_spaceex filesep filename '.xml']);

% instantiate system from converted SpaceEx model
sys_spaceex = feval(filename);


% instantiate equivalent CORA model

% top-left
inv = polytope([1 0; 0 -1],[0; 0]);
dynamics = linearSys([0 0; 0 0],[1; 0],[0; -1]);
% transitions
guard = polytope([0 -1],0,[-1 0],0);
reset = linearReset(eye(2),zeros(2,1),[10;5]);
trans(1) = transition(guard,reset,2);
guard = polytope([1 0],0,[0 1],0);
reset = linearReset(eye(2),zeros(2,1),[-5;-10]);
trans(2) = transition(guard,reset,3);
% define location
loc(1) = location('topleft',inv,trans,dynamics);

% top-right
inv = polytope([-1 0; 0 -1],[0; 0]);
dynamics = linearSys([0 0; 0 0],[0; 1],[-1; 0]);
% transitions
guard = polytope([0 -1],0,[1 0],0);
reset = linearReset(eye(2),zeros(2,1),[-10;5]);
trans(1) = transition(guard,reset,1);
guard = polytope([-1 0],0,[0 1],0);
reset = linearReset(eye(2),zeros(2,1),[5;-10]);
trans(2) = transition(guard,reset,4);
% define location
loc(2) = location('topright',inv,trans,dynamics);

% bottom-left
inv = polytope([1 0; 0 1],[0; 0]);
dynamics = linearSys([0 0; 0 0],[0; 1],[1; 0]);
% transitions
guard = polytope([1 0],0,[0 -1],0);
reset = linearReset(eye(2),zeros(2,1),[-5;10]);
trans(1) = transition(guard,reset,1);
guard = polytope([0 1],0,[-1 0],0);
reset = linearReset(eye(2),zeros(2,1),[10;-5]);
trans(2) = transition(guard,reset,4);
% define location
loc(3) = location('bottomleft',inv,trans,dynamics);

% bottom-right
inv = polytope([-1 0; 0 1],[0; 0]);
dynamics = linearSys([0 0; 0 0],[1; 0],[0; 1]);
% transitions
guard = polytope([-1 0],0,[0 -1],0);
reset = linearReset(eye(2),zeros(2,1),[5;10]);
trans(1) = transition(guard,reset,2);
guard = polytope([0 1],0,[1 0],0);
reset = linearReset(eye(2),zeros(2,1),[-10;-5]);
trans(2) = transition(guard,reset,3);
% define location
loc(4) = location('bottomright',inv,trans,dynamics);

% instantiate hybrid automaton
sys_cora = hybridAutomaton(loc);


% compare systems
assert(sys_cora == sys_spaceex);

% ------------------------------ END OF CODE ------------------------------
