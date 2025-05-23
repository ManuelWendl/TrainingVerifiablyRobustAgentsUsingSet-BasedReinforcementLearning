function res = testLong_spectraShadow_conZonotope
% testLong_spectraShadow_conZonotope - unit test function of conZonotope
%
% Syntax:
%    res = testLong_spectraShadow_conZonotope
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

% Authors:       Adrian Kulmburg
% Written:       05-April-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------


% empty spectrahedron
SpS_empty = spectraShadow.empty();
cZ_empty = conZonotope(SpS_empty);
assert(isemptyobject(cZ_empty));


% 1D, bounded, non-degenerate
SpS = spectraShadow([1 0 1 0;0 1 0 -1]);
cZ = conZonotope(SpS);
cZ_true = conZonotope(interval(-1,1));
assert(isequal(cZ,cZ_true,1e-10));

% 1D, empty
SpS = spectraShadow([-1 0]);
cZ = conZonotope(SpS);
cZ_true = conZonotope.empty(1);
assert(isequal(cZ,cZ_true,1e-10));

% 1D, single point
SpS = spectraShadow([-1 0 1 0;0 1 0 -1]);
cZ = conZonotope(SpS);
cZ_true = conZonotope(interval(1,1));
assert(isequal(cZ,cZ_true,1e-10));


% 2D, bounded, non-degenerate
A0 = eye(4);
Ai{1} = blkdiag([1 0;0 -1],zeros(2));
Ai{2} = blkdiag(zeros(2),[1 0;0 -1]);
SpS = spectraShadow([A0 Ai{1} Ai{2}]);
cZ = conZonotope(SpS);
cZ_true = conZonotope(interval([-1;-1],[1;1]));
assert(isequal(cZ,cZ_true,1e-10));

% 2D, bounded, degenerate
A0 = blkdiag([-1 0;0 1],[-1 0;0 1]);
Ai{1} = blkdiag([1 0;0 -1],zeros(2));
Ai{2} = blkdiag(zeros(2),[1 0;0 -1]);
SpS = spectraShadow([A0 Ai{1} Ai{2}]);
cZ = conZonotope(SpS);
cZ_true = conZonotope(interval([1;1],[1;1]));
assert(isequal(cZ,cZ_true,1e-10));

% 2D, empty
SpS = spectraShadow([-1 0 0]);
cZ = conZonotope(SpS);
cZ_true = conZonotope.empty(2);
assert(isequal(cZ,cZ_true,1e-10));


% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
