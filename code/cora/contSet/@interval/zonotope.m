function Z = zonotope(I)
% zonotope - Converts an interval object into a zonotope object
%
% Syntax:
%    Z = zonotope(I)
%
% Inputs:
%    I - interval object
%
% Outputs:
%    Z - zonotope object
%
% Example: 
%    I = interval([1;-1], [2; 1]);
%    Z = zonotope(I);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: interval, polytope

% Authors:       Matthias Althoff
% Written:       22-July-2016 
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% check if not a matrix set
n = dim(I);
if numel(n) > 1
    throw(CORAerror('CORA:wrongValue','first','Interval must not be an n-d array with n > 1.'))
end

% obtain center
c = center(I);

% construct generator matrix G
r = rad(I);
G = diag(r);

% instantiate zonotope
Z = zonotope([c,G(:,r ~= 0)]);

% ------------------------------ END OF CODE ------------------------------
