function res = test_ellipsoid_volume
% test_ellipsoid_volume - unit test function of volume
%
% Syntax:
%    res = test_ellipsoid_volume
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

% res = true;
assert((volume(ellipsoid.empty(2)) == 0));

% init cases
E1 = ellipsoid([ 5.4387811500952807 12.4977183618314545 ; 12.4977183618314545 29.6662117284481646 ], [ -0.7445068341257537 ; 3.5800647524843665 ], 0.000001);
Ed1 = ellipsoid([ 4.2533342807136076 0.6346400221575308 ; 0.6346400221575309 0.0946946398147988 ], [ -2.4653656883489115 ; 0.2717868749873985 ], 0.000001);
E0 = ellipsoid([ 0.0000000000000000 0.0000000000000000 ; 0.0000000000000000 0.0000000000000000 ], [ 1.0986933635979599 ; -1.9884387759871638 ], 0.000001);
n = dim(E1);
    
E1_vol = pi^(n/2)/gamma(n/2+1)*sqrt(det(E1.Q));

% check results
assert(withinTol(volume(E0),0,E0.TOL))
assert(withinTol(volume(Ed1),0,Ed1.TOL))
assert(withinTol(volume(E1),E1_vol,E1.TOL))

res = true;

% ------------------------------ END OF CODE ------------------------------
