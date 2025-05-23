function Gred = priv_lengthFilter(G,rem)
% priv_lengthFilter - filters out short generators
%
% Syntax:
%    Gred = priv_lengthFilter(G,rem)
%
% Inputs:
%    G - matrix of generators
%    rem - number of remaining generators
%
% Outputs:
%    Gred - reduced matrix of generators
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Matthias Althoff
% Written:       12-September-2008
% Last update:   14-March-2019 (norm and sort removed)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% pre-filter generators
h = vecnorm(G);

% choose largest values
[~,index]=maxk(h,rem);
Gred=G(:,index);

% ------------------------------ END OF CODE ------------------------------
