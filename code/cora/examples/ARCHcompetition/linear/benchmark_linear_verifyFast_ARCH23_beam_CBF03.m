function text = benchmark_linear_verifyFast_ARCH23_beam_CBF03
% benchmark_linear_verifyFast_ARCH23_beam_CBF03 - beam benchmark from the
%     2023 ARCH competition
%
% Syntax:
%    text = benchmark_linear_verifyFast_ARCH23_beam_CBF03()
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

% Model Derivation --------------------------------------------------------

% nodes in model
N = 1000;
% node of interest
node = round(0.7*N);

% constants
rho = 7.3e-4;   % density
L = 200;        % length of beam
Q = 1;          % cross-section area (renamed from A)
E = 30e6;       % Young's modulus

ell = L/N;      % length of individual discrete element

% mass matrix (NxN)
M = (rho*Q*ell) / 2 * diag([2*ones(N-1,1);1]);
Minv = M^(-1);

% load
F = zonotope(10000,100);

% tridiagonal matrix (NxN)
mat = zeros(N);
mat(1,1) = 2; mat(1,2) = -1;
mat(N,N-1) = -1; mat(N,N) = 1;
for r=2:N-1
    mat(r,1+(r-2)) = -1;
    mat(r,2+(r-2)) = 2;
    mat(r,3+(r-2)) = -1;
end
% stiffness matrix (NxN)
K = E*Q/ell * mat;
% damping matrix (NxN)
a = 1e-6;
b = 1e-6;
D = a*K + b*M;

% state matrix (damped)
A = [zeros(N) eye(N); -Minv*K -Minv*D];


% Parameters --------------------------------------------------------------

params.tFinal = 0.01;

% nr of states
dim_x = length(A);

% initial set: bar at rest
params.R0 = zonotope(zeros(dim_x,1));

% input set
params.U = cartProd( zonotope(zeros(dim_x-1,1)), Minv(end,end)*F );

options = struct();
options.verifyAlg = 'reachavoid:supportFunc';


% System Dynamics ---------------------------------------------------------

C = zeros(1,2*N);
C(1,2*node) = 1;

% construct linear system objects
sys = linearSys('beam',A,1,[],C);


% Specification -----------------------------------------------------------

% forall t: y1 <= 74
spec = specification(polytope(1,74),'safeSet');


% Verification ------------------------------------------------------------

% min steps needed: 5250
[res,fals,savedata] = verify(sys,params,options,spec);

disp("specifications verified: " + res);
disp("computation time: " + savedata.tComp);


% Return value ------------------------------------------------------------

text = ['Beam,CBF03,',num2str(res),',',num2str(savedata.tComp)];

% ------------------------------ END OF CODE ------------------------------
