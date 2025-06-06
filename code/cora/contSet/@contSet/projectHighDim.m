function S = projectHighDim(S,N,proj)
% projectHighDim - project a set to a higher-dimensional space,
%    having the new dimensions bounded at 0
%
% Syntax:
%    S = projectHighDim(S,N,proj)
%
% Inputs:
%    S - contSet object
%    N - dimension of the higher-dimensional space
%    proj - states of the high-dimensional space that correspond to the
%          states of the low-dimensional set
%
% Outputs:
%    S - contSet object in the higher-dimensional space
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: contSet/project, contSet/lift

% Authors:       Tobias Ladner
% Written:       13-September-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% parse input
narginchk(2,3);
if nargin < 3 || isempty(proj)
    proj = 1:dim(S);
end
inputArgsCheck({{S,'att','contSet'};
                {N,'att','numeric',{'nonnan','scalar','nonnegative','integer'}};
                {proj,'att','numeric',{'nonnan','vector','nonnegative'}}});
if dim(S) > N
    throw(CORAerror('CORA:wrongValue','second','Dimension of higher-dimensional space must be larger than or equal to the dimension of the given set.'))
elseif dim(S) ~= length(proj)
    throw(CORAerror('CORA:wrongValue','third','Number of dimensions in higher-dimensional space must match the dimension of the given set.'))
elseif max(proj) > N
    throw(CORAerror('CORA:wrongValue','third','Specified dimensions exceed dimension of high-dimensional space.'))
end

% call subfunction
S = projectHighDim_(S,N,proj);

% ------------------------------ END OF CODE ------------------------------
