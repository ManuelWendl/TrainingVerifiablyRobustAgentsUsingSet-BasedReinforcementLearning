function res = test_emptySet_project
% test_emptySet_project - unit test function of project
%
% Syntax:
%    res = test_emptySet_project
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
% See also: -

% Authors:       Mark Wetzlinger
% Written:       05-April-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% init empty set
n = 4;
O = emptySet(n);

% project to subspace
projDims = [1,3];
O_ = project(O,projDims);

% true solution
O_true = emptySet(length(projDims));

% compare solutions
assert(O_ == O_true);

% subspace out of range
projDims = [-1,2];
assertThrowsAs(@project,'CORA:outOfDomain',O,projDims);

% subspace out of range
projDims = [3,5];
assertThrowsAs(@project,'CORA:outOfDomain',O,projDims);

% test completed
res = true;

% ------------------------------ END OF CODE ------------------------------
