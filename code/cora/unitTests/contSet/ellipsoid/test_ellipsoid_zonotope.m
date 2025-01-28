function res = test_ellipsoid_zonotope
% test_ellipsoid_zonotope - unit test function of zonotope
%
% Syntax:
%    res = test_ellipsoid_zonotope
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

% Authors:       Victor Gassmann
% Written:       27-July-2021
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

res = true;

% init cases
E1 = ellipsoid([ 5.4387811500952807 12.4977183618314545 ; 12.4977183618314545 29.6662117284481646 ], [ -0.7445068341257537 ; 3.5800647524843665 ], 0.000001);
Ed1 = ellipsoid([ 4.2533342807136076 0.6346400221575308 ; 0.6346400221575309 0.0946946398147988 ], [ -2.4653656883489115 ; 0.2717868749873985 ], 0.000001);
E0 = ellipsoid([ 0.0000000000000000 0.0000000000000000 ; 0.0000000000000000 0.0000000000000000 ], [ 1.0986933635979599 ; -1.9884387759871638 ], 0.000001);
n = dim(E1);
N = 5*n;

Z1 = zonotope(E1,'inner:norm',2*n);
Zd1 = zonotope(Ed1,'outer:norm',2*n);
Z0 = zonotope(E0,'inner:norm_bnd',2*n);

assert(all(contains(E1,randPoint(Z1,N,'extreme'))))
assert(all(contains(Zd1,randPoint(Ed1,N,'extreme'))))
assert((all(rad(interval(Z0))==0) && all(withinTol(center(Z0),E0.q,E0.TOL))))

end

% ------------------------------ END OF CODE ------------------------------
