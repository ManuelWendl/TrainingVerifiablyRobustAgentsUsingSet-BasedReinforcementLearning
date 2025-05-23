function res = test_zonotope_representsa
% test_zonotope_representsa - unit test function of representsa
%
% Syntax:
%    res = test_zonotope_representsa
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

% Authors:       Mark Wetzlinger
% Written:       17-September-2019
% Last update:   ---
% Last revision: 20-July-2023 (MW, rename '...representsa')

% ------------------------------ BEGIN CODE -------------------------------

% 1. comparison to empty set
Z_empty = zonotope.empty(2);
assert(representsa(Z_empty,'emptySet'));

Z = zonotope([1, 2, 4;
               5, 6, 0;
              -1, 4, 8]);
assert(~representsa(Z,'emptySet'));


% 2. comparison to origin

% empty zonotope
Z = zonotope.empty(2);
assert(~representsa(Z,'origin'));

% only origin
Z = zonotope(zeros(3,1));
assert(representsa(Z,'origin'));

% shifted center
Z = zonotope(0.01*ones(4,1));
assert(~representsa(Z,'origin'));

% ...add tolerance
tol = 0.02;
assert(representsa(Z,'origin',tol));

% include generator matrix
Z = zonotope(ones(2,1),0.1*eye(2));
tol = 2;
assert(representsa(Z,'origin',tol));


% 3. comparison to interval
% create zonotopes
c1 = [0; 0];
G1 = [2 0; 0 1];
Z = zonotope(c1,G1);
[res,I] = representsa(Z,'interval');
assert(res);
assert(isequal(I,interval([-2;-1],[2;1])));

c2 = [1; 0];
G2 = [2 1; -1 4];
Z = zonotope(c2,G2);
assert(~representsa(Z,'interval'));


% 4. comparison to parallelotope
% check empty zonotope
Z = zonotope.empty(2);
assert(~representsa(Z,'parallelotope'));

% instantiate parallelotope
c = [-2; 1];
G = [2 4; -2 3];
Z = zonotope(c,G);
assert(representsa(Z,'parallelotope'));

% add zero-length generators
G = [G, zeros(2,2)];
Z = zonotope(c,G);
assert(representsa(Z,'parallelotope'));

% add generator -> not a parallelotope anymore
G = [G, [4; -2]];
Z = zonotope(c,G);
assert(~representsa(Z,'parallelotope'));

% no generator matrix
Z = zonotope(c);
assert(~representsa(Z,'parallelotope'));


% 5. comparison to point
Z = zonotope(ones(4,1));
assert(representsa(Z,'point'));
Z = zonotope([3;2;1],[0;0;eps]);
assert(representsa(Z,'point',1e-10));


% 6. comparison to more general classes
% always true and vice vers, too (in these cases!)
Z = zonotope([1;-1;2],[1 -3 0; 3 -1 1; -2 0 1]);
[res,S] = representsa(Z,'conZonotope');
assert(res);
assert(representsa(S,'zonotope'));
[res,S] = representsa(Z,'polyZonotope');
assert(res);
assert(representsa(S,'zonotope'));
% [res(end+1,1),S] = representsa(Z,'conPolyZono');
% res(end+1,1) = representsa(S,'zonotope');
% [res(end+1,1),S] = representsa(Z,'polytope');
% res(end+1,1) = representsa(S,'zonotope');


% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
