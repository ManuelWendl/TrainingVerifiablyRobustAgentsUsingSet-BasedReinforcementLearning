function res = minus(factor1, factor2)
% minus - Overloaded '-' operator for a Taylor model
%
% Syntax:
%    res = minus(factor1, factor2)
%
% Inputs:
%    factor1 and factor2 - a taylm objects
%    order  - the cut-off order of the Taylor series. The constat term is
%    the zero order.
%
% Outputs:
%    res - a taylm object
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: taylm, plus, mtimes
%
% References: 
%   [1] K. Makino et al. "Taylor Models and other validated functional 
%       inclusion methods"

% Authors:       Dmitry Grebenyuk
% Written:       20-April-2016
% Last update:   30-July-2017 (DG, multivariable polynomial pack is added)
%                02-December-2017 (DG, new rank evaluation)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

    if isscalar(factor1) && ~isscalar(factor2)
        res = arrayfun(@(b) aux_s_minus(factor1, b), factor2, 'UniformOutput', 0);
    elseif ~isscalar(factor1) && isscalar(factor2)
        res = arrayfun(@(a) aux_s_minus(a, factor2), factor1, 'UniformOutput', 0);  
    else
        res = arrayfun(@(a, b) aux_s_minus(a, b), factor1, factor2, 'UniformOutput', 0);  
    end
    A = cat(1, res{:});
    res = reshape(A, size(res));
    
end


% Auxiliary functions -----------------------------------------------------

% Implementation for a scalar
function res = aux_s_minus(factor1, factor2)

    if isa(factor1, 'taylm') && isa(factor2, 'taylm')

        % find the common variables 
        [factor1, factor2] = priv_rescale_dim(factor1, factor2);
        res = factor1;

        % Addition
        res.coefficients = [factor1.coefficients(:); -factor2.coefficients(:)];
        res.monomials = [factor1.monomials; factor2.monomials];
        
        % Merge the properties of the two taylor models
        res = priv_mergeProperties(res,factor1,factor2);

        % Reduce number of terms of the resulting Taylor model
        res = priv_compress(res);

        res.remainder = factor1.remainder - factor2.remainder;

        % if no polynomial part left, create an interval
        if isempty(res.coefficients) %subject to change. Maybe is wiser to keep the taylm
            res = res.remainder;
        end
    
    elseif isa(factor1,'taylm') && isa(factor2,'double')
        
        res = aux_substractConst(factor1,factor2);
        
    elseif isa(factor1,'double') && isa(factor2,'taylm')
        
        res = aux_substractFromConst(factor2,factor1);
     
    elseif isa(factor1,'taylm') && isa(factor2,'interval')
        
        res = factor1;
        res.remainder = res.remainder - factor2;
        
    elseif isa(factor1,'interval') && isa(factor2,'taylm') 
        
        res = -factor2;
        res.remainder = factor1 + res.remainder;
        
    else
        throw(CORAerror('CORA:wrongValue','first/second',...
            "be 'taylm ' or 'interval'"));
        
    end
end
    
function res = aux_substractConst(obj,const)

    res = obj;
        
    if isempty(obj.monomials) || obj.monomials(1,1) ~= 0
       res.coefficients = [-const;obj.coefficients];
       res.monomials = [zeros(1, length(names_of_var) + 1); obj.monomials];
    else
       res.coefficients(1) = obj.coefficients(1) - const; 
    end

end

function res = aux_substractFromConst(obj,const)

    res = obj;
        
    if isempty(obj.monomials) || obj.monomials(1,1) ~= 0
       res.coefficients = [const;-obj.coefficients];
       res.monomials = [zeros(1, length(names_of_var) + 1); obj.monomials];
    else
       res.coefficients = - obj.coefficients;
       res.coefficients(1) = res.coefficients(1) + const;
       res.remainder = - obj.remainder;
    end

end

% ------------------------------ END OF CODE ------------------------------
