function res = testMP_Krylov_iss_initReach
% testMP_Krylov_iss_initReach - unit test for checking the Krylov method
%    for the solution of the first time interval using the ISS model.
%    This test requires the multiple precision toolbox.
%
% Syntax:
%    res = testMP_Krylov_iss_initReach
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Matthias Althoff
% Written:       13-November-2018
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% enable access to private function "initReach_Krylov"
path = CORAROOT;
source1 = fullfile(path,'contDynamics','@linearSys','private','initReach_Krylov.m');
target1 = fullfile(path,'contDynamics','@linearSys','initReach_Krylov.m');
copyfile(source1,target1);
source2 = fullfile(path,'contDynamics','@linearSys','private','tie.m');
target2 = fullfile(path,'contDynamics','@linearSys','tie.m');
copyfile(source2,target2);
source3 = fullfile(path,'contDynamics','@linearSys','private','inputSolution.m');
target3 = fullfile(path,'contDynamics','@linearSys','inputSolution.m');
copyfile(source3,target3);
rmpath(genpath(path));
addpath(genpath(path));

% load system matrices
load('iss.mat');
n = length(A);

%set options --------------------------------------------------------------
R0 = interval(-0.0001*ones(n,1),0.0001*ones(n,1));
options.x0 = center(R0); %initial state for simulation

options.taylorTerms = 6;
options.zonotopeOrder = inf;
options.saveOrder = 1;
options.originContained = false;
options.reductionTechnique = 'girard';
options.linAlg = 'krylov';
options.compOutputSet = false;
options.saveOrder = 10;

U = interval([0;0.8;0.9],[0.1;1;1]);
options.U = zonotope([0;0;0],diag(rad(U)));
options.uTrans = center(U);

options.R0 = zonotope(R0);
options.tFinal = 20;
options.timeStep = 0.01;

options.krylovError = eps;
options.krylovStep = 20;
%--------------------------------------------------------------------------

%obtain factors for initial state and input solution
for i=1:(options.taylorTerms+1)
    %compute initial state factor
    options.factor(i)= options.timeStep^(i)/factorial(i);    
end

%specify continuous dynamics-----------------------------------------------
linDyn = linearSys('iss',A,B,[],1);
%--------------------------------------------------------------------------

% copy for Krylov techniques
linDyn_Krylov = linDyn;
options_Krylov = options;

% compute overapproximation using standard methods
[~, options] = initReach(linDyn, options.R0, options);

% compute overapproximation using Krylov methods
[~, options_Krylov] = initReach_Krylov(linDyn_Krylov, options_Krylov.R0, options_Krylov);

% check whether solutions of Krylov method enclose those of the standard 
% method; to save computational time, the results of the Krylov method are
% boxed
% homogeneous solution; time point
epsilonBox = 1e-7*interval(-ones(n,1),ones(n,1));
Rhom_tp_Krylov_boxed = interval(options_Krylov.Rhom_tp_proj);
Rhom_tp_boxed = interval(options.Rhom_tp);
assert(Rhom_tp_boxed <= (Rhom_tp_Krylov_boxed + epsilonBox));

% homogeneous solution; tie: tie can be more accurate for Krylov methods
Rtie_Krylov_boxed = interval(options_Krylov.R_tie_proj);
linDyn = tie(linDyn,options);
Rtie_boxed = interval(linDyn.taylor.F*zonotope(R0));
infDiff = abs(Rtie_boxed.inf - Rtie_Krylov_boxed.inf);
supDiff = abs(Rtie_boxed.sup - Rtie_Krylov_boxed.sup);
tieDiff = max(infDiff, supDiff);

% homogeneous solution; inputCorrection can be more accurate for Krylov methods
RinpCorr_Krylov_boxed = interval(options_Krylov.inputCorr_proj);
linDyn = inputSolution(linDyn,options);
RinpCorr_boxed = interval(linDyn.taylor.inputCorr);
infDiff = abs(RinpCorr_boxed.inf - RinpCorr_Krylov_boxed.inf);
supDiff = abs(RinpCorr_boxed.sup - RinpCorr_Krylov_boxed.sup);
inpDiff = max(infDiff, supDiff);

% homogeneous solution; time interval
Rhom_Krylov_boxed = interval(options_Krylov.Rhom_proj);
Rhom_boxed = interval(options.Rhom);
assert(Rhom_boxed <= (Rhom_Krylov_boxed + epsilonBox + interval(-(tieDiff+inpDiff),(tieDiff+inpDiff))));

% particulate solution
Raux_Krylov_boxed = interval(options_Krylov.Raux_proj);
Raux_boxed = interval(options.Raux);
assert(Raux_boxed <= (Raux_Krylov_boxed + epsilonBox));

% particulate solution of uTrans
Rtrans_Krylov_boxed = interval(options_Krylov.Rtrans_proj);
Rtrans_boxed = interval(options.Rtrans);
assert(Rtrans_boxed <= (Rtrans_Krylov_boxed + epsilonBox));

% delete copied functions
delete(target1);
delete(target2);
delete(target3);
rmpath(genpath(path));
addpath(genpath(path));

% test completed
res = true;

% ------------------------------ END OF CODE ------------------------------
