function res = test_polytope_setProperty_fullDim
% test_polytope_setProperty_fullDim - unit test function to check whether
%    the internally-used set property 'fullDim' is changed correctly
%    following different set operations on a polytope
%
% Syntax:
%    res = test_polytope_setProperty_fullDim
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
% See also: none

% Authors:       Mark Wetzlinger
% Written:       01-August-2023
% Last update:   ---
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% --- polytope ------------------------------------------------------------
% init set via vertex representation

% 1D, unbounded, non-degenerate
V = [-Inf, 2];
P = polytope(V);
assert(~isempty(P.fullDim.val) && P.fullDim.val);

% 1D, bounded, non-degenerate
V = [-3 -1 4 5];
P = polytope(V);
assert(~isempty(P.fullDim.val) && P.fullDim.val);

% 1D, bounded, degenerate
V = 2;
P = polytope(V);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);

% 2D, bounded, non-degenerate
V = [2 1; -1 4; -4 0; -1 -2; 3 -1]';
P = polytope(V);
assert(~isempty(P.fullDim.val) && P.fullDim.val);

% 2D, bounded, degenerate
V = [-1 1; 2 0]';
P = polytope(V);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% --- and -----------------------------------------------------------------

% 2D, bounded, non-degenerate & unbounded, degenerate
P1 = polytope([1 1; 1 -1; -1 0],[1;1;0.5]);
P2 = polytope(zeros(0,2),[],[0 1],0);
% determine degeneracy of P2
isFullDim(P2);
% compute intersection and check degeneracy of result
P = P1 & P2;
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% --- box -----------------------------------------------------------------

% 2D, empty set
P = polytope([1 0; -1 0],[2;-3]);
P_ = box(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);
assert(~isempty(P_.fullDim.val) && ~P_.fullDim.val);

% 2D, unbounded, non-degenerate, non-empty
P = polytope([1 0 0; 0 1 0],[1;-3]);
P_ = box(P);
assert(~isempty(P_.fullDim.val) && P_.fullDim.val);

% 2D, bounded, degenerate, non-empty
P = polytope([1 0 0; 0 1 0; -1 -1 0],[1;2;2],[0,0,1],0);
P_ = box(P);
assert(~isempty(P.emptySet.val) && ~P.emptySet.val);
assert(~isempty(P_.emptySet.val) && ~P_.emptySet.val);


% --- cartProd ------------------------------------------------------------

% 2D, bounded and numeric
P = polytope([1 1; -1 1; 0 -1],[1;1;1]);
S = 3;
P_ = cartProd(P,S);
assert(~isempty(P_.fullDim.val) && ~P_.fullDim.val);
P_ = cartProd(S,P);
assert(~isempty(P_.fullDim.val) && ~P_.fullDim.val);


% --- compact -------------------------------------------------------------

% 1D, empty
P = polytope([1;-1],[1;-2]);
P_ = compact(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val ...
    && ~isempty(P_.fullDim.val) && ~P_.fullDim.val);
% 3D, empty
P = polytope([1 1 0; -1 1 0; 0 -1 0; 1 0 0],[1;1;1;-3],[0 0 1],3);
P_ = compact(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val ...
    && ~isempty(P_.fullDim.val) && ~P_.fullDim.val);


% --- convHull ------------------------------------------------------------

% 2D, bounded, empty
P1 = polytope([1 1; -1 1; 0 -1],[1;1;1]);
P2 = polytope([0 1; -1 -1; 1 -1],[-1;0.1;0.1]);
P = convHull(P1,P2);
% P2 and P are empty
assert(~isempty(P2.fullDim.val) && ~P2.fullDim.val ...
    && ~isempty(P.fullDim.val) && ~P.fullDim.val);

% 1D, bounded, bounded
P1 = polytope([],[],1,2);
P2 = polytope([1;-1],[0;5]);
P = convHull(P1,P2);
% result is non-empty
assert(~isempty(P.fullDim.val) && P.fullDim.val);


% --- empty ---------------------------------------------------------------

n = 2;
P = polytope.empty(n);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% --- Inf -----------------------------------------------------------------

n = 2;
P = polytope.Inf(n);
assert(~isempty(P.fullDim.val) && P.fullDim.val);


% --- isFullDim -----------------------------------------------------------

% 2D, empty
P = polytope([0 1; -1 -1; 1 -1],[-1;0.1;0.1]);
isFullDim(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);

% 3D, full-dimensional
P = polytope([1 1 1; -1 1 1; 1 -1 1; 1 1 -1; -1 -1 1; -1 1 -1; 1 -1 -1; -1 -1 -1],ones(8,1));
isFullDim(P);
assert(~isempty(P.fullDim.val) && P.fullDim.val);

% 4D, degenerate
P = polytope([1 1 0 0; -1 1 0 0; 0 -1 0 0],[1;1;1],[0 0 2 0; 0 0 0 -2],[3;4]);
isFullDim(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% --- plus ----------------------------------------------------------------

% 2D, bounded + vector
A = [1 0; -1 1; -1 -1]; b = ones(3,1);
P = polytope(A,b);
isFullDim(P);
v = [-1;1];
P_sum = P + v;
% resulting polytope is also non-empty
assert(~isempty(P_sum.fullDim.val) && P_sum.fullDim.val);


% --- polytope ------------------------------------------------------------

% 2D, only inequalities, non-empty
P = polytope([1 1; -1 1; 0 -1],ones(3,1));
% determine degeneracy
isFullDim(P);
% copy polytope, property should also be copied
P_ = polytope(P);
assert(~isempty(P_.fullDim.val) && P_.fullDim.val);


% --- project ------------------------------------------------------------

% 3D, empty
P = polytope([1 0 0; -1 0 0],[2;-3]);
P_ = project(P,[1,2]);
assert(~isempty(P_.fullDim.val) && ~P_.fullDim.val);


% --- lift ----------------------------------------------------------------

% 2D, non-degenerate
P = polytope([1 1; -1 1; 0 -1],[1;1;1]);
% get knowledge about degeneracy
isFullDim(P);
% project to higher-dimensional space
P_ = lift(P,5,[2,3]);
% higher-dimensional polytope also non-empty
assert(~isempty(P_.fullDim.val) && P_.fullDim.val);

% 1D, degenerate
P = polytope([],[],1,-2);
% get knowledge about degeneracy
isFullDim(P);
% project to higher-dimensional space
P_ = lift(P,5,2);
% higher-dimensional polytope also non-empty
assert(~isempty(P_.fullDim.val) && ~P_.fullDim.val);


% --- representsa ---------------------------------------------------------

% 2D, origin
P = polytope([1 1; -1 1; 0 -1],zeros(3,1));
% determine degeneracy
representsa(P,'origin');
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);

% 1D, just a point
P = polytope([],[],1,2);
% determine degeneracy
representsa(P,'point');
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% --- vertices ------------------------------------------------------------

% 2D, empty set
P = polytope([1 0; -1 0],[2;-3]);
V = vertices(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);

% 2D, bounded, non-degenerate
P = polytope([1 0; 0 1; -1 -1],[2;1;2]);
V = vertices(P);
assert(~isempty(P.fullDim.val) && P.fullDim.val);

% 2D, bounded, degenerate
P = polytope([1 0; -1 0],[2;1],[0 1],3);
V = vertices(P);
assert(~isempty(P.fullDim.val) && ~P.fullDim.val);


% combine results
res = true;

% ------------------------------ END OF CODE ------------------------------
