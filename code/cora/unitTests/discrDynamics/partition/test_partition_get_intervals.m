function res = test_partition_get_intervals()
% test_partition_get_intervals - unit test cell intervals
%
% Syntax:
%    res = test_partition_get_intervals()
%
% Inputs:
%    no
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
oneDimField=partition([0,10],5);

% check that cellIntervals works, 1DOF
Ints1 = cellIntervals(oneDimField,1:nrOfCells(oneDimField));
Ints2 = cellIntervals(oneDimField);

assert(length(Ints1)==length(Ints2))
assert((norm(supremum(Ints1{3}) - supremum(Ints2{3}))<1e-9)&&(norm(infimum(Ints1{3}) - infimum(Ints2{3}))<1e-9));

% check that cellIntervals works, 2DOF
Ints1 = cellIntervals(twoDimField,1:nrOfCells(twoDimField));
Ints2 = cellIntervals(twoDimField);

assert(length(Ints1)==length(Ints2))
assert((norm(supremum(Ints1{3}) - supremum(Ints2{3}))<1e-9)&&(norm(infimum(Ints1{3}) - infimum(Ints2{3}))<1e-9));

% check that cellIntervals works, 3DOF
Ints1 = cellIntervals(threeDimField,1:nrOfCells(threeDimField));
Ints2 = cellIntervals(threeDimField);

assert(length(Ints1)==length(Ints2))
assert((norm(supremum(Ints1{3}) - supremum(Ints2{3}))<1e-9)&&(norm(infimum(Ints1{3}) - infimum(Ints2{3}))<1e-9));

res = true;

 
% segmentPolytope(threeDimField,[1 5 3])
% segmentPolytope(threeDimField)
% segmentZonotope(threeDimField,[1 5 3])
% segmentZonotope(threeDimField)
% P = polytope([2 0 0.3;4 2 0.6;1 1 0.5; 1 1 0.1]);
% intersectingSegments(threeDimField,P)
% [iS,percentages] = exactIntersectingCells(threeDimField,P)
% plot(threeDimField,exactIntersectingCells(threeDimField,P))
% hold on
% plot(P)
% 
% %partition with 

% ------------------------------ END OF CODE ------------------------------
