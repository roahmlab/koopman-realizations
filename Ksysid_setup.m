% Ksysid_setup
%
% Creates a sysid class and walks through all of the steps of building a
% model from data, validating its performance, and saving it (if desired)

clear Ksysid;

%% gather training data (need to prepare data file before running this)

% load in data file(s)
[ datafile_name , datafile_path ] = uigetfile( 'datafiles/*.mat' , 'Choose data file for sysid...' );
data4sysid = load( [datafile_path , datafile_name] );


%% construct sysid class
Ksysid = Ksysid( data4sysid ,...
        'model_type' , 'bilinear' ,...    % model type (linear, bilinear, or nonlinear)
        'time_type' , 'discrete' , ...  % 'discrete' or 'continuous'
        'obs_type' , { 'poly' } ,...    % type of basis functions
        'obs_degree' , [ 10 ] ,...       % "degree" of basis functions
        'snapshots' , Inf ,...          % Number of snapshot pairs
        'lasso' , [ Inf ] ,...            % L1 regularization term (Inf for least-squares sol.)
        'delays' , 0 ,...               % Numer of state/input delays
        'loaded' , false ,...           % Does system include external loads?
        'dim_red' , false);             % Should dimensional reduction be performed?

if Ksysid.loaded
    disp(['Number of basis functions: ' , num2str( (Ksysid.params.nw + 1) * Ksysid.params.N ) ]);
else
   disp(['Number of basis functions: ' , num2str( Ksysid.params.N ) ]);
end
    
%% basis dimensional reduction (no longer needed, baked into class constructor)

% disp('Performing dimensional reduction...');
% Px = Ksysid.lift_snapshots( Ksysid.snapshotPairs );
% Ksysid = Ksysid.get_econ_observables( Px );
% disp(['Number of basis functions: ' , num2str( Ksysid.params.N ) ]);
% clear Px;
    
%% train model(s)
Ksysid = Ksysid.train_models;


%% validate model(s)
% could also manually do this for one model at a time

results = cell( size(Ksysid.candidates) );    % store results in a cell array
err = cell( size(Ksysid.candidates) );    % store error in a cell array 

if iscell(Ksysid.candidates)
    for i = 1 : length(Ksysid.candidates)
        [ results{i} , err{i} ] = Ksysid.valNplot_model( i );
    end
else
    [ results{1} , err{1} ] = Ksysid.valNplot_model;
end
    
% calculate aggregate error accross all trials
toterr.mean = zeros( size(err{1}{1}.mean) );
toterr.rmse = zeros( size(err{1}{1}.rmse) );
toterr.nrmse = zeros( size(err{1}{1}.nrmse) );
for i = 1:length(err{1})
    toterr.mean = toterr.mean + err{1}{i}.mean; 
    toterr.rmse = toterr.rmse + err{1}{i}.rmse;
    toterr.nrmse = toterr.nrmse + err{1}{i}.nrmse;
end

%% save model(s)

% You do this based on the validation results.
% Call this function:
%   Ksysid.save_class( )


