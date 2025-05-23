function res = testLong_linearSysDT_observe_08_pedestrian()
% testLong_linearSysDT_observe_08_pedestrian - unit_test_function for
%    guaranteed state estimation of linear discrete-time systems.
%
%    Checks the solution of the linearSysDT class for a pedestrian example;
%    It is checked whether the enclosing interval of the final observed set 
%    is close to an interval provided by a previous (saved) solution
%
% Syntax:
%    res = testLong_linearSysDT_observe_08_pedestrian
%
% Inputs:
%    -
%
% Outputs:
%    res - true/false 

% Authors:       Matthias Althoff
% Written:       03-March-2021
% Last update:   31-January-2023 (TL, updated IH_saved for FRad-C in >R2022a)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------
 
% assume true
res = true;


% Load pedestrian model ---------------------------------------------------
load pedestrianModel pedestrian params options simRes


% Set of evaluated estimators
Estimator = {
    %'VolMin-B' % exclude for time saving of unit test
    'FRad-A' 
    'FRad-B' 
    %'PRad-A' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'PRad-B' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'PRad-C' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'PRad-D' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'CZN-A' % exclude for time saving of unit test
    %'CZN-B' % exclude for time saving of unit test
    'ESO-A'
    'ESO-B'
    'FRad-C' 
    %'PRad-E' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'Nom-G' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'ESO-C' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'ESO-D' % tested in testSDPT3_linearSysDT_observe_08_pedestrian()
    %'Hinf-G' % tested in testSDPT3_linearSysDT_observe_08_pedestrian() 
    };

% init
resPartial = [];

% set tolerance
tol = 1e-6;

% perform evaluation ------------------------------------------------------
% loop over estimators
for iEst = 4:length(Estimator)

    % set algorithm
    estName = Estimator{iEst};
    options.alg = estName;
    
    % if constrained zonotopes or ellipsoids are used
    paramsNew = params;
    if any(strcmp(estName,{'ESO-A','ESO-B','ESO-C','ESO-D','ESO-E'}))
        paramsNew.R0 = ellipsoid(params.R0);
        paramsNew.W = ellipsoid(params.W);
        paramsNew.V = ellipsoid(params.V);
    elseif any(strcmp(estName,{'CZN-A','CZN-B'}))
        paramsNew.R0 = conZonotope(params.R0);
    end
    
    % evaluate observer
    estSet = observe(pedestrian,paramsNew,options);

    % enclose last estimated set by interval
    IH = interval(estSet.timePoint.set{end});
    
    % obtain enclosing intervals
    switch options.alg
        case 'VolMin-A'
            IH_saved = [];
        case 'VolMin-B' 
            IH_saved = []; 
        case 'FRad-A'
            IH_saved = interval( ...
                [0.9689629614880655; 2.0431176164148965; -0.9053334051832218; -0.2439026226497167], ...
                [1.4009520199423995; 2.4748897477268015; 1.0721434684190747; 1.7016428573164182]);
        case 'FRad-B' 
            IH_saved = interval( ...
                [0.9706379673448939; 2.0484523710670945; -0.9448902873388959; -0.2574323446416942], ...
                [1.3986790926035191; 2.4698120103720980; 1.1112964875172127; 1.7190120616864819]);
        case 'PRad-A' 
            IH_saved = [];
        case 'PRad-B' 
            IH_saved = [];
        case 'PRad-C' 
            IH_saved = [];
        case 'FRad-C' 
            % TL: numerically unstable...
            IH_saved = interval( ...
                [ 0.9592223696721490; 2.0268025113919803; -0.9290244580855602; -0.3052542874485443 ], ...
                [ 1.4089942984695809; 2.4885219977302739; 1.0908607144881430; 1.7537703992533271 ]);
        case 'PRad-D' 
            IH_saved = [];
        case 'PRad-E' 
            IH_saved = [];
        case 'Nom-G' 
            IH_saved = []; 
        case 'Hinf-G' 
            IH_saved = [];        
        case 'ESO-A'
            IH_saved = interval( ...
                [-6.2161992351739261; -6.1042993440301068; -4.3916499693186024; -4.2479105199518692], ...
                [8.3043471808976737; 8.4162470720414539; 4.7941281805365916; 4.9378676299033089]);
        case 'ESO-B'
            IH_saved = interval( ...
                [-7.7805372634866066; -7.4993040804409610; -6.3205268323980528; -6.1025379971956379], ...
                [10.1058864645813280; 10.3871196476269780; 6.8127620432373250; 7.0307508784397417]);
        case 'ESO-C'
            IH_saved = [];   
        case 'ESO-D'
            IH_saved = [];
        case 'CZN-A'
            IH_saved = [];
        case 'CZN-B'
            IH_saved = [];
    end

    % check equality with tolerance
    assert(isequal(IH,IH_saved,tol));
end

% final result
res = all(resPartial);

% ------------------------------ END OF CODE ------------------------------
