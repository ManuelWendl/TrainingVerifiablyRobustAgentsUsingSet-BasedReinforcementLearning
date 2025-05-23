function res = test_contSet_times
% test_contSet_times - unit test function of times
%
% Syntax:
%    res = test_contSet_times
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

% test interval
I = interval([1;3],[2;5]);
M = [3;-1];
MI = M .* I;
assert(all(MI.inf == [3;-5] & MI.sup == [6;-3], 'all'));

% test empty case
I = interval.empty(2);
I = 2 .* I;
assert(representsa_(I,'emptySet',eps));

% test zonotope
Z = zonotope([1 0; 2 1]);
M = [-1;2];
MZ = M .* Z;
assert(all([MZ.c,MZ.G] == [-1 0; 4 2], 'all'));

res = true;

% ------------------------------ END OF CODE ------------------------------
