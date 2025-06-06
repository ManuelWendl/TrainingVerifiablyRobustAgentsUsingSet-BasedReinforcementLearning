function res = test_partition_get_zonotopes()
% test_partition_get_zonotopes - unit test cell zonotopes
%
% Syntax:
%    res = test_partition_get_zonotopes()
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false

% Authors:       Aaron Pereira, Matthias Althoff
% Written:       02-August-2017
% Last update:   02-August-2018 (MA)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------


%setup partitions
threeDimField=partition([0,10; -3,3; 0,1],[5;10;3]);
twoDimField=partition([0,10; -3,3],[5;10]);

% check that cellZonotopes works, 3DOF
Zons1 = cellZonotopes(twoDimField,1:nrOfCells(twoDimField));
Zons2 = cellZonotopes(twoDimField);

assert(length(Zons1)==length(Zons2))
assert((norm(center(Zons1{3}) - center(Zons2{3}))<1e-9)&&(norm(generators(Zons1{3}) - generators(Zons2{3}))<1e-9));

% check that cellZonotopes works, 3DOF
Zons1 = cellZonotopes(threeDimField,1:nrOfCells(threeDimField));
Zons2 = cellZonotopes(threeDimField);

assert(length(Zons1)==length(Zons2))
assert((norm(center(Zons1{3}) - center(Zons2{3}))<1e-9)&&(norm(generators(Zons1{3}) - generators(Zons2{3}))<1e-9));

res = true;

% segmentPolytope(threeDimField,[1 5 3])
% segmentPolytope(threeDimField)
% cellZonotopes(threeDimField,[1 5 3])
% cellZonotopes(threeDimField)
% P = polytope([2 0 0.3;4 2 0.6;1 1 0.5; 1 1 0.1]);
% intersectingSegments(threeDimField,P)
% [iS,percentages] = exactIntersectingCells(threeDimField,P)
% plot(threeDimField,exactIntersectingCells(threeDimField,P))
% hold on
% plot(P)
% 
% %partition with 

% ------------------------------ END OF CODE ------------------------------
