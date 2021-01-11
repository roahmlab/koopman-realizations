% evaluate_rand-system_models
% 
%   Trains linear/bilinear/nonlinear models of random systems
%   Validates sytems models against data
%   Quantifies average error in validation trials
%   Plots the average normalized error for each model type/degree

%% Set training parameters and choose data

% save models?
save_models = false;

% Maximum degree for each type of model
max_degree_linear = 20;
max_degree_bilinear = 10;
max_degree_nonlinear = 5;

% load in data file(s)
[ datafile_name , datafile_path ] = uigetfile( 'datafiles/*.mat' , 'Choose data file for sysid...' );
load( [datafile_path , datafile_name] );    % should be called 'data4sysid_all'

%% Train models for each random system

% make directories for saving models
if save_models
    folder_all_models = [ 'systems' , filesep , data4sysid_all{1}.folder_name ];
    folder_linear_models = [ folder_all_models , filesep , 'linear' ];
    folder_bilinear_models = [ folder_all_models , filesep , 'bilinear' ];
    folder_nonlinear_models = [ folder_all_models , filesep , 'nonlinear' ];
    
    % create directories for storing models
    mkdir( folder_all_models );
    mkdir( folder_linear_models );
    mkdir( folder_bilinear_models );
    mkdir( folder_nonlinear_models );
end

err_linear_models = zeros( max_degree_linear , size( data4sysid_all , 1 ) );  % preallocate
err_bilinear_models = zeros( max_degree_bilinear , size( data4sysid_all , 1 ) );  % preallocate
err_nonlinear_models = zeros( max_degree_nonlinear , size( data4sysid_all , 1 ) );  % preallocate
dim_linear_models = zeros( max_degree_linear , size( data4sysid_all , 1 ) );  % preallocate
dim_bilinear_models = zeros( max_degree_bilinear , size( data4sysid_all , 1 ) );  % preallocate
dim_nonlinear_models = zeros( max_degree_nonlinear , size( data4sysid_all , 1 ) );  % preallocate

for i = 1 : size( data4sysid_all , 1 )
    % train linear models
    for j = 1 : max_degree_linear
        clear Ksysid;
        Ksysid = Ksysid( data4sysid_all{i} ,...
            'model_type' , 'linear' ,...    % model type (linear, bilinear, or nonlinear)
            'time_type' , 'discrete' , ...  % 'discrete' or 'continuous'
            'obs_type' , { 'poly' } ,...    % type of basis functions
            'obs_degree' , [ j ] ,...       % "degree" of basis functions
            'snapshots' , Inf ,...          % Number of snapshot pairs
            'lasso' , [ Inf ] ,...            % L1 regularization term (Inf for least-squares sol.)
            'delays' , 0 ,...               % Numer of state/input delays
            'loaded' , false ,...           % Does system include external loads?
            'dim_red' , false);             % Should dimensional reduction be performed?
        Ksysid = Ksysid.train_models;
        
        % save the model
        if save_models
            model = Ksysid.model;
            file_name = [ Ksysid.model_type , '_' , Ksysid.obs_type{1} , '-' , num2str(Ksysid.obs_degree) ,'_n-' , num2str( Ksysid.params.n ) , '_m-' , num2str( Ksysid.params.m ) , '_del-' , num2str( Ksysid.params.nd ) ];
            save( [ folder_linear_models , filesep , file_name ] , '-struct' , 'model' );
        end
        
        % validate model
        results = Ksysid.valNplot_model([],false,false);
        mean_error = results{1}.error.mean;  % mean error over all time steps
        mean_error_zeros = sum( abs( results{1}.real.y ) ) / size(results{1}.real.y,1); % mean error over all time steps, zero response
        normed_mean_error = results{1}.error.mean / mean_error_zeros;   % mean error normed by zero response
        
        % put all the normed mean error for linear models in single matrix
        err_linear_models( j , i ) = normed_mean_error;
        dim_linear_models( j , i ) = size( Ksysid.basis.full , 1 );
    end
    
    % train bilinear models
    for j = 1 : max_degree_bilinear
        clear Ksysid;
        Ksysid = Ksysid( data4sysid_all{i} ,...
            'model_type' , 'bilinear' ,...    % model type (linear, bilinear, or nonlinear)
            'time_type' , 'discrete' , ...  % 'discrete' or 'continuous'
            'obs_type' , { 'poly' } ,...    % type of basis functions
            'obs_degree' , [ j ] ,...       % "degree" of basis functions
            'snapshots' , Inf ,...          % Number of snapshot pairs
            'lasso' , [ Inf ] ,...            % L1 regularization term (Inf for least-squares sol.)
            'delays' , 0 ,...               % Numer of state/input delays
            'loaded' , false ,...           % Does system include external loads?
            'dim_red' , false);             % Should dimensional reduction be performed?
        Ksysid = Ksysid.train_models;
        
        % save the model
        if save_models
            model = Ksysid.model;
            file_name = [ Ksysid.model_type , '_' , Ksysid.obs_type{1} , '-' , num2str(Ksysid.obs_degree) ,'_n-' , num2str( Ksysid.params.n ) , '_m-' , num2str( Ksysid.params.m ) , '_del-' , num2str( Ksysid.params.nd ) ];
            save( [ folder_bilinear_models , filesep , file_name ] , '-struct' , 'model' );
        end
        
        % validate model
        results = Ksysid.valNplot_model([],false,false);
        mean_error = results{1}.error.mean;  % mean error over all time steps
        mean_error_zeros = sum( abs( results{1}.real.y ) ) / size(results{1}.real.y,1); % mean error over all time steps, zero response
        normed_mean_error = results{1}.error.mean / mean_error_zeros;   % mean error normed by zero response
        
        % put all the normed mean error for linear models in single matrix
        err_bilinear_models( j , i ) = normed_mean_error;
        dim_bilinear_models( j , i ) = size( Ksysid.basis.full_input , 1 );
    end
    
    % train nonlinear models
    for j = 1 : max_degree_nonlinear
        clear Ksysid;
        Ksysid = Ksysid( data4sysid_all{i} ,...
            'model_type' , 'nonlinear' ,...    % model type (linear, bilinear, or nonlinear)
            'time_type' , 'discrete' , ...  % 'discrete' or 'continuous'
            'obs_type' , { 'poly' } ,...    % type of basis functions
            'obs_degree' , [ j ] ,...       % "degree" of basis functions
            'snapshots' , Inf ,...          % Number of snapshot pairs
            'lasso' , [ Inf ] ,...            % L1 regularization term (Inf for least-squares sol.)
            'delays' , 0 ,...               % Numer of state/input delays
            'loaded' , false ,...           % Does system include external loads?
            'dim_red' , false);             % Should dimensional reduction be performed?
        Ksysid = Ksysid.train_models;
        
        % save the model
        if save_models
            model = Ksysid.model;
            file_name = [ Ksysid.model_type , '_' , Ksysid.obs_type{1} , '-' , num2str(Ksysid.obs_degree) ,'_n-' , num2str( Ksysid.params.n ) , '_m-' , num2str( Ksysid.params.m ) , '_del-' , num2str( Ksysid.params.nd ) ];
            save( [ folder_nonlinear_models , filesep , file_name ] , '-struct' , 'model' );
        end
        
        % validate model
        results = Ksysid.valNplot_model([],false,false);
        mean_error = results{1}.error.mean;  % mean error over all time steps
        mean_error_zeros = sum( abs( results{1}.real.y ) ) / size(results{1}.real.y,1); % mean error over all time steps, zero response
        normed_mean_error = results{1}.error.mean / mean_error_zeros;   % mean error normed by zero response
        
        % put all the normed mean error for linear models in single matrix
        err_nonlinear_models( j , i ) = normed_mean_error;
        dim_nonlinear_models( j , i ) = size( Ksysid.basis.full , 1 );
    end
end

%% Plot the error

% linear
mean_linear = mean( err_linear_models , 2 );
std_linear = std( err_linear_models' )';

% bilinear
mean_bilinear = mean( err_bilinear_models , 2 );
std_bilinear = std( err_bilinear_models' )';

% nonlinear
mean_nonlinear = mean( err_nonlinear_models , 2 );
std_nonlinear = std( err_nonlinear_models' )';

% plot the comparison (need to edit figure to make it look nice
std_scale = 0.25; % reduce scale of standard devation for better viewing
figure;
hold on;
errorbar( dim_linear_models(:,1) , mean_linear , std_scale * std_linear , 'b' );
errorbar( dim_bilinear_models(:,1) , mean_bilinear , std_scale * std_bilinear , 'g' );
errorbar( dim_nonlinear_models(:,1) , mean_nonlinear , std_scale * std_nonlinear , 'r' );
hold off;
xlabel('Numer of basis functions')
ylabel('Normalized error')
legend( 'linear' , 'bilinear' , 'nonlinear' )
ylim([0,1]);












