function res = test_matZonotope_intervalMatrix
% test_matZonotope_intervalMatrix - unit test function for conversion to
%    intervalMatrix
% 
% Syntax:
%    res = test_matZonotope_intervalMatrix
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
% Written:       26-April-2024
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% empty matrix zonotope
matZempty = matZonotope();
intMat = intervalMatrix(matZempty);
assert(representsa(intMat,'emptySet'));

% scalar
C = 0;
G = []; G(:,:,1) = 1; G(:,:,2) = -2;
matZscalar = matZonotope(C,G);
intMat = intervalMatrix(matZscalar);
assert(representsa(intMat,'interval'));

% nx1 vector
C = [0; 1; 1];
G = []; G(:,:,1) = [1; -1; -2]; G(:,:,2) = [-2; 0; 1];
matZvector = matZonotope(C,G);
intMat = intervalMatrix(matZvector);
assert(representsa(intMat,'interval'));

% matrix
C = [0 2; 1 -1; 1 -2];
G = []; G(:,:,1) = [1 1; -1 0; -2 1]; G(:,:,2) = [-2 0; 0 1; 1 -1];
matZ = matZonotope(C,G);
intMat = intervalMatrix(matZ);
assert(representsa(intMat,'interval'));

% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
