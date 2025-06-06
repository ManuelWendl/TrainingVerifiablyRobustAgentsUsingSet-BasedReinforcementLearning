function HA = rendezvous_SRA01()
% rendezvous_SRA01 - linear spacecraft-rendezvous benchmark, see [1, Sec. 3.2]
%
% Syntax:  
%    HA = rendezvous_SRA01()
%
% Inputs:
%    -
%
% Outputs:
%    HA - hybridAutomaton object
% 
% References:
%    [1] M. Althoff, “ARCH-COMP19 Category Report: Continuous and Hybrid 
%        Systems with Linear Continuous Dynamics", 2019
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:        Niklas Kochdumper
% Written:        22-May-2020
% Last update:    ---
% Last revision:  ---

% ------------------------------ BEGIN CODE -------------------------------


%% Generated on 20-Apr-2018

%----------Automaton created from Component 'ChaserSpacecraft'-------------

%% Interface Specification:
%   This section clarifies the meaning of state & input dimensions
%   by showing their mapping to SpaceEx variable names. 

% Component 1 (ChaserSpacecraft):
%  state x := [x; y; vx; vy; t]
%  input u := [uDummy]

%----------------------Component ChaserSpacecraft--------------------------

%-------------------------------State P2-----------------------------------

%% equation:
%   t'==1 & x'==vx & y'==vy & vx'==-0.057599765881773*x+0.000200959896519766*y-2.89995083970656*vx+0.00877200894463775*vy & vy'==-0.000174031357370456*x-0.0665123984901026*y-0.00875351105536225*vx-2.90300269286856*vy
dynA = ...
[0,0,1,0,0;0,0,0,1,0;-0.05759976588,0.0002009598965,-2.89995084,...
0.008772008945,0;-0.0001740313574,-0.06651239849,-0.008753511055,...
-2.903002693,0;0,0,0,0,0];
dynB = ...
[0;0;0;0;0];
dync = ...
[0;0;0;0;1];
dynamics = linearSys('linearSys', dynA, dynB, dync);

%% equation:
%   t<=125&x<=-100
invA = ...
[0,0,0,0,1;1,0,0,0,0];
invb = ...
[125;-100];
inv = polytope(invA, invb);

trans = transition();
%% equation:
%   
resetA = ...
[1,0,0,0,0;0,1,0,0,0;0,0,1,0,0;0,0,0,1,0;0,0,0,0,1];
resetc = ...
[0;0;0;0;0];
reset = linearReset(resetA,[],resetc);

%% equation:
%   y>=-100 & x+y >=-141.1 & x>=-100 & y-x<=141.1 & y<=100 & x+y<=141.1 & x<=100 & y-x>=-141.1
guardA = [-1,0,0,0,0];
guardb = 100;
guard = polytope([],[],guardA,guardb);

trans = transition(guard, reset, 2);

%% equation:
%   
resetA = ...
[1,0,0,0,0;0,1,0,0,0;0,0,1,0,0;0,0,0,1,0;0,0,0,0,1];
resetc = ...
[0;0;0;0;0];
reset = linearReset(resetA,[],resetc);

%% equation:
%   t>=120
guardA = ...
[0,0,0,0,-1];
guardb = ...
[-120];
guard = polytope([],[],guardA,guardb);

trans(2) = transition(guard, reset, 3);

loc(1) = location('S1', inv, trans, dynamics);


%-------------------------------State P3-----------------------------------

%% equation:
%   t'==1 & x'==vx & y'==vy & vx'==-0.575999943070835*x+0.000262486079431672*y-19.2299795908647*vx+0.00876275931760007*vy & vy'==-0.000262486080737868*x-0.575999940191886*y-0.00876276068239993*vx-19.2299765959399*vy
dynA = ...
[0,0,1,0,0;0,0,0,1,0;-0.5759999431,0.0002624860794,-19.22997959,...
0.008762759318,0;-0.0002624860807,-0.5759999402,-0.008762760682,...
-19.2299766,0;0,0,0,0,0];
dynB = ...
[0;0;0;0;0];
dync = ...
[0;0;0;0;1];
dynamics = linearSys('linearSys', dynA, dynB, dync);

%% equation:
%   t<=125 & y>=-100 & x+y>=-141.1 & x>=-100 & y-x<=141.1 & y<=100 & x+y<=141.1 & x<=100 & y-x>=-141.1
invA = ...
[0,0,0,0,1;0,-1,0,0,0;-1,-1,0,0,0;-1,0,0,0,0;-1,1,0,0,0;0,1,0,0,0;1,1,0,...
0,0;1,0,0,0,0;1,-1,0,0,0];
invb = ...
[125;100;141.1;100;141.1;100;141.1;100;141.1];
inv = polytope(invA, invb);

trans = transition();
%% equation:
%   
resetA = ...
[1,0,0,0,0;0,1,0,0,0;0,0,1,0,0;0,0,0,1,0;0,0,0,0,1];
resetc = ...
[0;0;0;0;0];
reset = linearReset(resetA,[],resetc);

%% equation:
%   t>=120
guardA = ...
[0,0,0,0,-1];
guardb = ...
[-120];
guard = polytope([],[],guardA,guardb);

trans = transition(guard, reset, 3);

loc(2) = location('S2', inv, trans, dynamics);


%-----------------------------State Passive--------------------------------

%% equation:
%   t'==1 & x'==vx & y'==vy & vx'==0.0000575894721132000*x+0.00876276*vy & vy'==-0.00876276*vx 
dynA = ...
[0,0,1,0,0;0,0,0,1,0;5.758947211E-05,0,0,0.00876276,0;0,0,-0.00876276,0,...
0;0,0,0,0,0];
dynB = ...
[0;0;0;0;0];
dync = ...
[0;0;0;0;1];
dynamics = linearSys('linearSys', dynA, dynB, dync);

%% equation:
%   t>=120
invA = ...
[0,0,0,0,-1];
invb = ...
[-120];
inv = polytope(invA, invb);

trans = transition();
loc(3) = location('S3', inv, trans, dynamics);


HA = hybridAutomaton('rendezvous_SRA01',loc);

% ------------------------------ END OF CODE ------------------------------