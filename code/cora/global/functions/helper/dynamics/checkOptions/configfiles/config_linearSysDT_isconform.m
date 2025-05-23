function [paramsList,optionsList] = config_linearSysDT_isconform
% config_linearSysDT_isconform - configuration file for validation of
%    model parameters and algorithm parameters
%
% Syntax:
%    [paramsList,optionsList] = config_linearSysDT_isconform
%
% Inputs:
%    -
%
% Outputs:
%    paramsList - list of model parameters
%    optionsList - list of algorithm parameters
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: add2list

% Authors:       Tobias Ladner
% Written:       05-October-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% init structs
[paramsList,optionsList] = initDynParameterList();

% list of model parameters ------------------------------------------------

% mandatory
paramsList(end+1,1) = add2list('R0','mandatory');
paramsList(end+1,1) = add2list('tFinal','mandatory');
paramsList(end+1,1) = add2list('testSuite','mandatory');

% default
paramsList(end+1,1) = add2list('tStart','default');
paramsList(end+1,1) = add2list('U','default');
paramsList(end+1,1) = add2list('u','default');
paramsList(end+1,1) = add2list('W','default');
paramsList(end+1,1) = add2list('V','default');

% optional

% list of algorithm parameters --------------------------------------------

% default
optionsList(end+1,1) = add2list('zonotopeOrder','default');
optionsList(end+1,1) = add2list('verbose','default');
optionsList(end+1,1) = add2list('reductionTechnique','default');
optionsList(end+1,1) = add2list('compOutputSet','default');
optionsList(end+1,1) = add2list('reachAlg','default');
optionsList(end+1,1) = add2list('timeStepDivider','default');
optionsList(end+1,1) = add2list('postProcessingOrder','default');

% optional
optionsList(end+1,1) = add2list('saveOrder','optional');

end


% Auxiliary functions -----------------------------------------------------

function list = aux_filter(list,removeList)
    idx = cellfun(@(x) ~ismember(x,removeList), {list.name},'UniformOutput',true);
    list = list(idx);

end

% ------------------------------ END OF CODE ------------------------------
