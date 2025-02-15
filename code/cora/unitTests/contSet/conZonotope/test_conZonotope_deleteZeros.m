function res = test_conZonotope_deleteZeros
% test_conZonotope_deleteZeros - unit test function of deleteZeros
%
% Syntax:
%    res = test_conZonotope_deleteZeros
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
% Written:       09-January-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% empty conZonotope
cZ_empty = conZonotope.empty(2);
cZ_empty_ = deleteZeros(cZ_empty);
assert(representsa_(cZ_empty_,'emptySet',eps));

% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
