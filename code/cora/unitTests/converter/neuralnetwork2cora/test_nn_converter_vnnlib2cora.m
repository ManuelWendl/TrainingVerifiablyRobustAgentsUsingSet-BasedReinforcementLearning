function res = test_nn_converter_vnnlib2cora()
% test_nn_converter_vnnlib2cora - unit test for 
%    converting vnnlib files from the VNN competition to cora
%
% Syntax:
%    res = test_nn_converter_vnnlib2cora()
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false 
%

% Authors:       Tobias Ladner
% Written:       30-August-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% try to load different vnnlib files with interesting structure

% mnistfc benchmark
[X0, spec] = vnnlib2cora('mnistfc_prop_11_0.05.vnnlib');
assert(dim(X0{1}) == 28*28 && all(size(spec.set.A) == [9,10]));

% axas_xu benchmark
[X0, spec] = vnnlib2cora('axas_xu_prop_3.vnnlib');
assert(dim(X0{1}) == 5 && all(size(spec.set.A) == [4,5]));

% rl_learning benchmark
[X0, spec] = vnnlib2cora('rl_benchmark_dubinsrejoin_case_unsafe_11.vnnlib');
assert(dim(X0{1}) == 8 && all(size(spec.set.A) == [6,8]));

% gather results
res = true;

% ------------------------------ END OF CODE ------------------------------
