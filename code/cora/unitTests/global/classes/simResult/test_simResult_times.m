function res = test_simResult_times
% test_simResult_times - unit test function for times
%
% Syntax:
%    res = test_simResult_times()
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
% Written:       02-February-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% init simResult object
n = 3;
N = 100;

x = {rand(N,n),rand(N,n)};
t = {(1:N)',(1:N)'};

simRes = simResult(x,t);

% simple cases
A = [2; 3; -1];
simRes_out = A .* simRes;
assert(isequal(simRes_out.x{1},x{1}.*A') ...
    && isequal(simRes_out.x{2},x{2}.*A'));

A = 3;
simRes_out = A .* simRes;
assert(isequal(simRes_out.x{1},x{1}*A') ...
    && isequal(simRes_out.x{2},x{2}*A'));

A = 3;
simRes_out = simRes .* A;
assert(isequal(simRes_out.x{1},x{1}*A') ...
    && isequal(simRes_out.x{2},x{2}*A'));

% gather results
res = true;

% ------------------------------ END OF CODE ------------------------------
