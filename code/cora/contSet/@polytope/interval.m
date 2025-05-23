function I = interval(P)
% interval - encloses a polytope by an interval
%
% Syntax:
%    I = interval(P)
%
% Inputs:
%    P - polytope object
%
% Outputs:
%    I - interval 
%
% Example:
%    A = [1 2; -1 1; -1 -3; 2 -1];
%    b = ones(4,1);
%    P = polytope(A,b);
%    
%    I = interval(P);
%
%    figure; hold on;
%    plot(P);
%    plot(I,[1,2],'k');
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Authors:       Matthias Althoff, Viktor Kotsev, Mark Wetzlinger
% Written:       01-February-2011
% Last update:   30-July-2016
%                31-May-2022
%                14-December-2022 (MW, simplification)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% dimension
n = dim(P);

% obtain bounding box in halfspace representation
if P.isVRep.val
    [A,b,empty] = priv_box_V(P.V_.val,n);
else
    [A,b,empty] = priv_box_H(P.A_.val,P.b_.val,P.Ae_.val,P.be_.val,n);
end

% exit if already empty
if empty
    I = interval.empty(n);
    return
end

% init lower and upper bounds of resulting interval with Inf values
lb = -Inf(n,1);
ub = Inf(n,1);

% indices of constraints for upper bounds (indices of constraints for lower
% bounds are given by the logical opposite)
idx_ub = any(A > 0,2);
nnz_ub = nnz(idx_ub);

% upper bounds that are non-Inf
idx_nonInf = any(A(idx_ub,:),1);

% overwrite bounds using b
ub(idx_nonInf) = b(1:nnz_ub);

% lower bounds that are non-(-Inf)
idx_nonInf = any(A(~idx_ub,:),1);

% overwrite bounds using b
lb(idx_nonInf) = -b(nnz_ub+1:end);

% instantiate resulting interval
I = interval(lb,ub);

% ------------------------------ END OF CODE ------------------------------
