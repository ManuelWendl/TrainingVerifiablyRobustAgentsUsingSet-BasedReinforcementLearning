function res = test_zonotope_uplus
% test_zonotope_uplus - unit test function of uminus
%
% Syntax:
%    res = test_zonotope_uplus
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

% Authors:       Tobias Ladner
% Written:       06-April-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% init
c = [0;0];
G = [2 0 2; 0 2 2];
Z = zonotope(c, G);

% plus
pZ = Z;
assert(all([pZ.c,pZ.G] == [c,G], 'all'));

% compare with Z
assert(isequal(pZ, Z));

% test empty case
assert(isemptyobject(+zonotope.empty(2)));

% add results
res = true;

% ------------------------------ END OF CODE ------------------------------
