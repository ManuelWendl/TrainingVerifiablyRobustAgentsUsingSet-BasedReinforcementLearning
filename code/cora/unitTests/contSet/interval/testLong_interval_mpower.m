function res = testLong_interval_mpower
% testLong_interval_mpower - unit_test_function of power,
%    overloaded '^' operator for intervals
%
% Syntax:
%    res = testLong_interval_mpower
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
% See also: mtimes

% Authors:       Dmitry Grebenyuk, Mark Wetzlinger
% Written:       05-January-2016
% Last update:   08-August-2020 (MW, extend by random tests)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

res = true;

% Random tests ---------------------------------------------------------

numberRandTests = 10000;

% if exponent = 0, inf and sup are 1
for i=1:numberRandTests
    a = interval(-randi(1000,1,1)*rand(1),randi(1000,1,1)*rand(1));
    b = a^0;
    assertLoop(b.inf == 1,i)
    assertLoop(b.sup == 1,i)
end

% if exponent is even, inf is greater or equal to than 0
for i=1:numberRandTests
    a = interval(-randi(1000,1,1)*rand(1),randi(1000,1,1)*rand(1));
    exponent = 2*randi([1,4],1,1); % 2,4,6,8
    b = a^exponent;
    assertLoop(b.inf >= 0,i)
end

% if inf > 0, inf after operation also > 0
for i=1:numberRandTests
    randinf = randi(1000,1,1)*rand(1);
    a = interval(randinf,randinf*2);
    exponent = randi([1,5],1,1);
    b = a^exponent;
    assertLoop(b.inf > 0,i)
end

% if exponent is non-integer, run through only if inf > 0
for i=1:numberRandTests
    randinf = randi(1000,1,1)*rand(1);
    a = interval(randinf,randinf*2);
    
    b = a^rand(1);
    assertLoop(b.inf > 0,i)
end

for i=1:numberRandTests
    a = interval(-randi(1000,1,1)*rand(1),randi(1000,1,1)*rand(1));
    try
        b = a^rand(1); % should throw an error
        % in case no error thrown, result should be NaN
        assertLoop(all(isnan(b.inf)),i)
    catch
        continue
    end
end

% ------------------------------ END OF CODE ------------------------------
