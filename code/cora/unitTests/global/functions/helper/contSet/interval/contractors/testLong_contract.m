function res = testLong_contract()
% testLong_contract - unit test function for contractors
%
% Syntax:
%    res = testLong_contract()
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
% See also: contract, contractPoly

% Authors:       Niklas Kochdumper
% Written:       18-December-2020
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

tol = 1e-10;

% Test 1: Different Contractors -------------------------------------------

% test if all contractors that are implemented provide a valid result for
% contraction

% contraction problem
f = @(x) x(1)^2 + x(2)^2 - 4;
dom = interval([1;1],[3;3]);

% optimal result
Iopt = interval([1;1],[sqrt(3);sqrt(3)]);

% contractors
cont = {'forwardBackward','polynomial','linearize','interval','all'};
iter = 2;
splits = 2;

% loop over all contractors
for i = 1:length(cont)
    
    % contract the domain
    I1 = contract(f,dom,cont{i},iter,splits);
    assertLoop(contains(I1,Iopt,'exact',tol),i);
end


% Test 2: "contract" vs. "contractPoly" -----------------------------------

% test if "contract" and "contractPoly" give the same result if they are
% called for the same contraction problem

% contraction problem
f = @(x) x(1)^2 + x(2)^2 - 4;
dom = interval([1;1],[3;3]);

% equivalent formulation with a polynomial function
c = -4; G = [1 1]; GI = []; E = 2*eye(2);

% contractors
cont = {'forwardBackward','polynomial','linearize','interval','all'};
iter = 2;
splits = 2;

% loop over all contractors
for i = 1:length(cont)
    
    % contract with "contract"
    I1 = contract(f,dom,cont{i},iter,splits);
    
    % contract with "contractPoly"
    I2 = contractPoly(c,G,GI,E,dom,cont{i},iter,splits);
    
    % check the result for correctness
    assertLoop(isequal(I1,I2,tol),i);
end


% Test 3: Multiple Constraints --------------------------------------------

% contraction problem
f = @(x) [x(1)^2 - 4*x(2); 
               x(2)^2 - 2*x(1) + 4*x(2)];
dom = interval([-0.1;-0.1],[0.1;0.1]);

% equivalent formulation with a polynomial function
c = [0;0]; G = [1 0 0 -4;0 1 -2 4]; E = [2 0 1 0;0 2 0 1]; GI = [];

% contractors
cont = {'forwardBackward','polynomial','linearize','interval','all'};
iter = 2;
splits = 2;

% loop over all contractors
for i = 1:length(cont)
    
    % contract with "contract"
    I1 = contract(f,dom,cont{i},iter,splits);
    
    % contract with "contractPoly"
    I2 = contractPoly(c,G,GI,E,dom,cont{i},iter,splits);
    
    % check the result for correctness
    assertLoop(isequal(I1,I2,tol),i)
end

res = true;

% ------------------------------ END OF CODE ------------------------------
