function res = test_capsule_dim
% test_capsule_dim - unit test function of dim
%
% Syntax:
%    res = test_capsule_dim
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

% Authors:       Mark Wetzlinger
% Written:       27-September-2019
% Last update:   12-March-2021 (MW, add empty case)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% empty case
n = 2;
C = capsule.empty(n);
assert(dim(C) == n);

% 2D capsule
c = [1;1]; g = [1;1]; r = 0.5;
C = capsule(c);
assert(dim(C) == 2);
C = capsule(c,g,r);
assert(dim(C) == 2);


% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
