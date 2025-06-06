function res = test_nn_taylm_sigmoid()
% test_nn_taylm_sigmoid - tests nn with sigmoid activation using taylor models 
%
% Syntax:
%    res = test_nn_taylm_sigmoid()
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

% Authors:       Tobias Ladner
% Written:       24-June-2022
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------
 
% assume true
res = true;

% load W, b, input_ref, output_ref (from previous model)
model = "model_test_nn_taylm_taylor_sigmoid.mat";
load(model)

% build the new layer based model
layers = {};
for i = 1:length(W)
    % concat layers
    activation_layer = nnActivationLayer.instantiateFromString(options.nn.activation);
    layers = [; ...
        layers; ...
        {nnLinearLayer(W{i}, b{i})}; ...
        {activation_layer}; ...
        ];
end
nn_new = neuralNetwork(layers);

% calculate output
options.nn.poly_method = 'taylor';
output_new = nn_new.evaluate(input_ref, options);

assert(isequal(interval(output_ref), interval(output_new), 1e-15));

end

% ------------------------------ END OF CODE ------------------------------
