function res = mrdivide(numerator,denominator)
% mrdivide - Overload '/' operator for Taylor models
%
% Syntax:
%    res = mrdivide(numerator,denominator)
%
% Inputs:
%    numerator - numerator (class taylm)
%    denominator - denominator (class taylm)
%
% Outputs:
%    res - resulting taylm object
%
% Other m-files required: priv_minverse
% Subfunctions: none
% MAT-files required: none
%
% See also: taylm

% Authors:       Dmitry Grebenyuk
% Written:       14-June-2017
% Last update:   ---  
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

if isscalar(denominator)
    res = numerator ./ denominator;
else
    res = numerator * priv_minverse(denominator);
end

% ------------------------------ END OF CODE ------------------------------
