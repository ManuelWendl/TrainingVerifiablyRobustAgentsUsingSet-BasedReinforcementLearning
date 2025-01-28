function text = benchmark_linear_verifyFast_ARCH23_iss_ISSC01_ISS02
% benchmark_linear_verifyFast_ARCH23_iss_ISSC01_ISS02 - iss benchmark from
%     the 2023 ARCH competition
%
% Syntax:
%    text = benchmark_linear_verifyFast_ARCH23_iss_ISSC01_ISS02()
%
% Inputs:
%    -
%
% Outputs:
%    text - char array

% Authors:       Mark Wetzlinger
% Written:       23-March-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Parameters --------------------------------------------------------------

R0 = [interval(-0.0001*ones(270,1),0.0001*ones(270,1)); ...
      interval([0;0.8;0.9],[0.1;1;1])];
params.U = zonotope(0);
params.R0 = zonotope(R0);
params.tFinal = 20;

options = struct();
options.verifyAlg = 'reachavoid:supportFunc';


% Specifications ----------------------------------------------------------

% forall t: -5e-4 <= y3 <= 5e-4
d = 5e-4;
P1 = polytope([0 0 1],-d);
P2 = polytope([0 0 -1],-d);
spec = specification({P1,P2},'unsafeSet');


% System Dynamics ---------------------------------------------------------

% load system matrices
load iss.mat A B C

% construct extended system matrices (inputs as additional states)
dim_x = length(A);
A_  = [A,B;zeros(size(B,2),dim_x + size(B,2))];
B_  = zeros(dim_x+size(B,2),1);
C_  = [C,zeros(size(C,1),size(B,2))];

% construct the linear system object
sys = linearSys('iss',A_,B_,[],C_);


% Verification ------------------------------------------------------------

% min steps needed: 185
[res,fals,savedata] = verify(sys,params,options,spec);

disp("specifications verified: " + res);
disp("computation time: " + savedata.tComp);

% Return value ------------------------------------------------------------

text = ['Spacestation,ISSC01-ISS02,',num2str(res),',',num2str(savedata.tComp)];

% ------------------------------ END OF CODE ------------------------------
