function res = test_partition()
% test_partition - unit test for the partition class and the functions 
% cellSegments and cellIndices.
%
% Syntax:
%    res = test_partition()
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

% test of the partition class and the functions cellSegments and cellIndices

%one dimensional case
oneDimField=partition([0,10],5);
assert(~any(cellSegments(oneDimField,[1 3 4]) - [1 3 4]'));
assert(~any(cellIndices(oneDimField,[1 3 4]') - [1 3 4]));

%two dimensional case
twoDimField=partition([0,10; -3,3],[5;10]);

subs = cellSegments(twoDimField,[37    26    30]);
assert(norm(subs - [2,8;1,6;5,6])<1e-15);

inds = cellIndices(twoDimField,[2,8;1,6;5,6]);
assert(norm(inds - [37    26    30])<1e-15);

subs = cellSegments(twoDimField,[34    46   109   0   -2  51  -10]);
inds = cellIndices(twoDimField,subs);
assert(norm(inds - [34    46   0   0   0  0  0]) < 1e-15);


% three dimensional case
threeDimField=partition([0,10; -3,3; 0,1],[5;10;3]);

% index to subscripts
subs = cellSegments(threeDimField,[87    46   109]);
assert(norm(subs - [2,8,2;1,10,1;4,2,3]) < 1e-15);

% subscripts to indices
inds = cellIndices(threeDimField,[2,8,2;1,10,1;4,2,3]);
assert(norm(inds - [87    46   109]) < 1e-15);

% does it also work when out of bounds??
subs = cellSegments(threeDimField,[87    46   109   0   150  151  2000  -10]);
inds = cellIndices(threeDimField,subs);
assert(norm(inds - [87    46   109     0   150     0     0     0]) < 1e-15);


% new definition of partition
threeDimField_div=partition({[0 2 3 4 8 10],[-3 -1.5 -1 -0.9 0 0.1 0.2 0.3 1 2 3],[0,0.3,0.6,1]});

% did the new definition work?
assert(norm(threeDimField_div.nrOfSegments - threeDimField.nrOfSegments) < 1e-15);

% index to subscripts
subs = cellSegments(threeDimField_div,[87    46   109]);
assert(norm(subs - [2,8,2;1,10,1;4,2,3]) < 1e-15);

% subscripts to indices
inds = cellIndices(threeDimField_div,[2,8,2;1,10,1;4,2,3]);
assert(norm(inds - [87    46   109]) < 1e-15);

% does it work when out of bounds??
subs = cellSegments(threeDimField_div,[87    46   109   0   150  151  2000  -10]);
inds = cellIndices(threeDimField_div,subs);
assert(norm(inds - [87    46   109     0   150     0     0     0]) < 1e-15);

res = true;

% ------------------------------ END OF CODE ------------------------------
