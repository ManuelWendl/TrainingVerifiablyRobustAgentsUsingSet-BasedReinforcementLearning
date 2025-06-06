function obj = priv_mergeProperties(obj,inp1,inp2)
% priv_mergeProperties - Merge the object properties of two taylm objects
%
% Syntax:
%    obj = priv_mergeProperties(obj,inp1,inp2)
%
% Inputs:
%    obj - taylm object whos properties get modified
%    inp1, inp2 - taylm objects for which the properties get merged
%
% Outputs:
%    obj - taylm object
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: taylm

% Authors:       Niklas Kochdumper
% Written:       06-April-2018
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

obj.max_order = max(inp1.max_order,inp2.max_order);
obj.opt_method = inp1.opt_method;
obj.eps = min(inp1.eps,inp2.eps);
obj.tolerance = min(inp1.tolerance,inp2.tolerance);

% ------------------------------ END OF CODE ------------------------------
