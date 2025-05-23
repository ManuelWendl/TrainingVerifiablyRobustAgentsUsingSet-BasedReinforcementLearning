function res = priv_isFinalLocation(loc,finalLoc)
% priv_isFinalLocation - checks if given location is final location
%
% Syntax:
%    res = priv_isFinalLocation(loc,finalLoc)
%
% Inputs:
%    loc - number of current location
%    finalLoc - vector of final locations
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
% Written:       ---
% Last update:   16-June-2022 (MW, simplify entire function)
%                15-October-2024 (MW, rename)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% check if current location is equal to any of the possible final locations
% note: if no final location specified, finalLoc = 0 which results in false
% as loc is always greater than zero
res = any(loc == finalLoc);

% ------------------------------ END OF CODE ------------------------------
