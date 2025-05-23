classdef emptySet < contSet
% emptySet - object constructor for empty sets
%
% Description:
%    This class represents empty sets.
%
% Syntax:
%    obj = emptySet(n)
%
% Inputs:
%    n - dimension
%
% Outputs:
%    obj - generated emptySet object
%
% Example: 
%    n = 2;
%    O = emptySet(n);
%    plot(O);
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: ---

% Authors:       Mark Wetzlinger
% Written:       22-March-2023
% Last update:   ---
% Last revision: 10-January-2024 (MW, reformat)

% ------------------------------ BEGIN CODE -------------------------------

properties (SetAccess = {?contSet, ?matrixSet}, GetAccess = public)
    % dimension of space
    dimension = 0;
end

methods

    function obj = emptySet(varargin)
        
        % 0. avoid empty instantiation
        if nargin == 0
            throw(CORAerror('CORA:noInputInSetConstructor'));
        end
        assertNarginConstructor(1,nargin);

        % 1. copy constructor
        if nargin == 1 && isa(varargin{1},'emptySet')
            obj = varargin{1};
            return;
        end

        % 2. parse input arguments
        n = varargin{1};

        % 3. check correctness of input arguments
        inputArgsCheck({{n,'att','numeric',{'scalar','nonnegative','integer'}}});
        
        % 4. assign properties
        obj.dimension = n;

        % 5. set precedence (fixed)
        obj.precedence = 0;
    end
end

methods (Static = true)
    O = generateRandom(varargin) % generate random empty set
    O = empty(n) % instantiates an empty empty set
end

methods (Access = protected)
    [abbrev,printOrder] = getPrintSetInfo(S)
end

end

% ------------------------------ END OF CODE ------------------------------
