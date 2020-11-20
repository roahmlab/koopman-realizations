classdef Ksysid
    %Ksysid: Class for doing Koopman-based sysid
    %   Constructed from a sample set of data from an experiment or simulation.
    
    properties
        params struct;  % sysid parameters
        lift struct;    % contains matlab generated lifting functions
        basis struct;   % contains the observables for the system
        
        model;   % contains the bese lifted system model of system (chosen from candidates)
        
        candidates; % candidate models trained with different lasso params
        koopData;   % info related to the training of koopman operator
        
        scaledown;      % contains scaling functions for y,u,zeta to lie in [-1,1]
        scaleup;        % contains scaling from [-1,1] to real life range
        
        % properties that can be set using Name,Value pair input to constructor
        isupdate;   % true if this should overwrite an existing model when saving, false otherwise
        obs_type;   % cell array of the types of observables to include in model
        obs_degree; % array of the degree/complexity of each type of observable
        snapshots;  % number of snapshot pairs to use in training
        lasso;      % lasso L1 regularization penalty weight 
        delays;     % number of delays to include
        liftinput;  % if 1, input is lifted along with state, yielding nonlinear model
        model_type; % 'linear' or 'nonlinear'
        loaded;     % true or false. Does the system include loads?
        dim_red;    % true or false. Should SVD dimensional reduction be performed?
        time_type;  % 'discrete' or 'continuous'
        
        traindata;  % scaled exp/sim data for training the model
        valdata;    % scaled exp/sim data for validating the model
        snapshotPairs;  % snapshot pairs extracted from training data
    end
    
    methods
        function obj = Ksysid( data4sysid , varargin )
            %CLASS CONSTRUCTOR
            %   data - struct with fields t,x,u,y. Contains results of a
            %   simuation or experiment of the system to be identified
            %   (note: this is not necessarily the same data to be used for
            %   sysid, it just provides the dimensions of the state, input,
            %   etc.)
            
            % verify that data4sysid has required fields
            if ~isfield( data4sysid , 'train' ) || ~isfield( data4sysid , 'val' )
                error('Input must have *train* and *val* fields of type cell array');
            end
            
            % isolate one trial to extract some model parameters
            data = data4sysid.train{1};
            data4train = data4sysid.train;
            data4val = data4sysid.val;
            
            % set param values based on the data
            obj.params = struct;    % initialize params struct
            obj.params.n = size( data.y , 2 );  % dimension of measured state
            obj.params.m = size( data.u , 2 );  % dimension of input
            obj.params.Ts = mean( data.t(2:end) - data.t(1:end-1) );    % sampling time
            obj.params.isfake = false;  % assume system is real
            
            % if data has a params field save it as sysParams
            if isfield( data , 'params' )
                obj.params.sysParams = data.params;
                obj.params.isfake = true;   % if params field exist the system is fake
            end
            
            % initialize structs
            obj.lift = struct;  % initialize lift struct
            obj.basis = struct; % initialize basis struct
            
            % set defualt values of Name, Value optional arguments
            obj.isupdate = false;
            obj.obs_type = { 'poly' };
            obj.obs_degree = [ 1 ];
            obj.snapshots = Inf;
            obj.lasso = [ 1e6 ]; % default is least squares solution
            obj.delays = 0; % default 0 (was 1 to ensure model is dynamic)
            obj.model_type = 'linear';
            obj.loaded = false;
            obj.time_type = 'discrete';
            
            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );
            obj.params.nd = obj.delays;  % saves copy in params (for consistency with mpc class)
            obj.params.nzeta = obj.params.n * ( obj.delays + 1 ) + obj.params.m * obj.delays;
            
            % specify dimension of the loading condition
            if obj.loaded && isfield( data , 'w' )
                obj.params.nw = size( data.w , 2 );
            else
                obj.params.nw = 0;
            end
            
            % verify specified model type is valid
            if strcmp( obj.model_type , 'linear' )
                obj.liftinput = 0;  % default is not to lift input
            elseif strcmp( obj.model_type , 'nonlinear' )
                obj.liftinput = 1;  % lift the input
            elseif strcmp( obj.model_type , 'bilinear' )
                obj.liftinput = 2;  % lift the input (try to avoid using this parameter)
            else
                error('Invalid model_type chosen. Must be linear, bilinear, or nonlinear.');
            end    
            
            % make sure if user specifies a "loaded" model the data has a load field (w)
            if obj.loaded && ~isfield( data , 'w' )
                error('You have specified a loaded system, but your training data does not have the required load field (w)');
            end
        
            % define the set of observables
            if obj.loaded   % observables must include loads
                obj = obj.def_observables_loaded( obj.obs_type , obj.obs_degree );
            else
                obj = obj.def_observables( obj.obs_type , obj.obs_degree );
            end
            
            % merge the training data into a single big file (requred for training function to work)
            data4train_merged = obj.merge_trials( data4train );
            
            % scale data to be in range [-1 , 1]
            [ traindata , obj ] = obj.get_scale( data4train_merged );
            valdata = cell( size( data4val ) );
            for i = 1 : length( data4val )
                valdata{i} = obj.scale_data( data4val{i} );
            end
            obj.traindata = traindata;
            obj.valdata = valdata;
%             % To suppress scaling, comment this in and comment out above 2 lines
%             obj.traindata = data4train_merged;
%             obj.valdata = data4val;
            
            % get shapshot pairs from traindata
            obj.snapshotPairs = obj.get_snapshotPairs( obj.traindata , obj.snapshots );
        end
        
        % parse_args: Parses the Name, Value pairs in varargin
        function obj = parse_args( obj , varargin )
            %parse_args: Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.(varargin{idx}) = varargin{idx+1} ;
            end
            
            % If lasso is set to Inf, replace it with a valid large number
            if obj.lasso == Inf
                obj.lasso = 1e6;
            end
        end
        
        % loop_progress: visually display the progress of a for loop
        function loop_progress( obj, now , total )
             %loop_progress: display progress in execution of for loop
             
             prog = now / total;
             numdots = floor( prog * 10 );
             
             if prog <= 0.1     % empty progress bar
                fprintf(['Loop progress: [' , repmat('.',1,10) , ']\n']);
             elseif prog >= 0.9     % last progress bar
                fprintf(repmat('\b',1,28));
                fprintf(['Loop progress: [' , repmat('#',1,10) , repmat('.',1,10-10) , ']\n']);
             else
                 fprintf(repmat('\b',1,28));
                 fprintf(['Loop progress: [' , repmat('#',1,numdots) , repmat('.',1,10-numdots) , ']\n']);
             end
        end
        
        %% operations on simulation/experimental data (some are redundant and found in the data class)
        
        function [ data_scaled , obj ] = get_scale( obj , data )
            %scale: Scale sim/exp data to be in range [-1 , 1]
            %    Also creates scaleup/scaledown matrices and saves as params
            %    data - struct containing fields t , y , u (at least)
            %    data_scaled - struct containing t , y , u , x (optional)   
            
            % get min/max values in each dimension
            y_min = min( data.y );
            u_min = min( data.u );
            y_max = max( data.y );
            u_max = max( data.u );
            
            % calculate centers of range
            y_dc = ( y_max + y_min ) ./ 2;
            u_dc = ( u_max + u_min ) ./ 2;
            
            % calculate scaling factors
            scale_y = ( y_max - y_min ) ./ 2;
            scale_u = ( u_max - u_min ) ./ 2;
            
            % shift and scale the data
            data_scaled = struct;    % initialize
            data_scaled.t = data.t;  % time is not scaled
            data_scaled.y = ( data.y - y_dc ) ./ scale_y;
            data_scaled.u = ( data.u - u_dc ) ./ scale_u;
            
            % save scaling functions
            y = sym( 'y' , [ 1 , obj.params.n ] );
            u = sym( 'u' , [ 1 , obj.params.m ] );
            y_scaledown = ( y - y_dc ) ./ scale_y;
            u_scaledown = ( u - u_dc ) ./ scale_u;
            obj.scaledown.y = matlabFunction( y_scaledown , 'Vars' , {y} );
            obj.scaledown.u = matlabFunction( u_scaledown , 'Vars' , {u} );
            
            y_scaleup = ( y .* scale_y ) + y_dc;
            u_scaleup = ( u .* scale_u ) + u_dc;
            obj.scaleup.y = matlabFunction( y_scaleup , 'Vars' , {y} );
            obj.scaleup.u = matlabFunction( u_scaleup , 'Vars' , {u} );
            
            % save scaling factors
            obj.params.scale.y_factor = scale_y;
            obj.params.scale.y_offset = y_dc;
            obj.params.scale.u_factor = scale_u;
            obj.params.scale.u_offset = u_dc;
            
            % do same for x if it is part of data struct
            if ismember( 'x' , fields(data) )
                x_min = min( data.x );
                x_max = max( data.x );
                x_dc = ( x_max + x_min ) ./ 2;
                scale_x = ( x_max - x_min ) ./ 2;
                data_scaled.x = ( data.x - x_dc ) ./ scale_x;
                x = sym( 'x' , [ 1 , size(data.x,2) ] );
                x_scaledown = ( x - x_dc ) ./ scale_x;
                obj.scaledown.x = matlabFunction( x_scaledown , 'Vars' , {x} );
                x_scaleup = ( x .* scale_x ) + x_dc;
                obj.scaleup.x = matlabFunction( x_scaleup , 'Vars' , {x} );
            end
            
            % do same for w if it is part of data struct
            if ismember( 'w' , fields(data) )
                w = sym( 'w' , [ 1 , size(data.w,2) ] );
                w_min = min( data.w );
                w_max = max( data.w );
                w_dc = ( w_max + w_min ) ./ 2;
                w_nonconst_ind = find( w_min ~= w_max );   % find non-constant load parameters
                % if load is constant just shift to zero
                w_scaledown = w - w_dc;
                w_scaleup = w + w_dc;
                % otherwise shift and scale it between [-1,1]
                for i = w_nonconst_ind
                    scale_w = ( w_max(i) - w_min(i) ) ./ 2;
                    w_scaledown(i) = ( w(i) - w_dc(i) ) ./ scale_w;
                    w_scaleup(i) = ( w(i) .* scale_w ) + w_dc(i);
                end
                obj.scaledown.w = matlabFunction( w_scaledown , 'Vars' , {w} );
                obj.scaleup.w = matlabFunction( w_scaleup , 'Vars' , {w} );
                data_scaled.w = obj.scaledown.w( data.w );
            end
            
            % create scaling functions for zeta
            zeta = sym( 'zeta' , [ 1 , obj.params.nzeta ] );
            zeta_scaledown = sym( zeros( size(zeta) ) );
            zeta_scaleup = sym( zeros( size(zeta) ) );
            zeta_scaledown(1:obj.params.n) = obj.scaledown.y( zeta(1:obj.params.n) );
            zeta_scaleup(1:obj.params.n) = obj.scaleup.y( zeta(1:obj.params.n) );
            for i = 1 : obj.delays   % for y delays
                range = obj.params.n * i + 1 : obj.params.n * (i+1);
                zeta_scaledown(range) = obj.scaledown.y( zeta(range) );
                zeta_scaleup(range) = obj.scaleup.y( zeta(range) );
            end
            for i = 1 : obj.delays   % for u delays
                endy = obj.params.n * ( obj.delays + 1 );
                range = endy + obj.params.m * (i-1) + 1 : endy + obj.params.m * i;
                zeta_scaledown(range) = obj.scaledown.u( zeta(range) );
                zeta_scaleup(range) = obj.scaleup.u( zeta(range) );
            end
            obj.scaledown.zeta = matlabFunction( zeta_scaledown , 'Vars' , {zeta} );
            obj.scaleup.zeta = matlabFunction( zeta_scaleup , 'Vars' , {zeta} );
        end
        
        % resample (resamples data with a desired time step)
        function data_resampled = resample( obj , data , Ts )
            %resample: resamples sim/exp data with a desired timestep
            %   data - struct with fields t, y, x (optional)
            %   Ts - the desired sampling period
            
            % get query points
            tq = ( data.t(1) : Ts : data.t(end) )';
            
            data_resampled.t = tq;
            data_resampled.u = interp1( data.t , data.u , tq );
            data_resampled.y = interp1( data.t , data.y , tq );
            if ismember( 'x' , fields(data) )
                data_resampled.x = interp1( data.t , data.x , tq );
            end
            if ismember( 'w' , fields(data) )
                data_resampled.w = interp1( data.t , data.w , tq );
            end
        end
        
        % scale_data (scale sim/exp data to be in range [-1 , 1])
        function data_scaled = scale_data( obj , data , down )
            %scale: Scale sim/exp data based on the scalefactors set in
            % get_scale.
            %    data - struct containing fields t , y , u (at least)
            %    data_scaled - struct containing t , y , u , x (optional)
            %    down - boolean. true to scale down, false to scale up.
            
            if nargin < 3
                down = true; % default is to scale down
            end
            
            % scale the data
            if down
                data_scaled = struct;    % initialize
                data_scaled.t = data.t;  % time is not scaled
                data_scaled.y = obj.scaledown.y( data.y );  %data.y * obj.params.scaledown.y;
                data_scaled.u = obj.scaledown.u( data.u );  %data.u * obj.params.scaledown.u;
                if ismember( 'x' , fields(data) )
                    data_scaled.x = obj.scaledown.x( data.x );  %data.x * obj.params.scaledown.x;
                end
                if ismember( 'w' , fields(data) )
                    data_scaled.w = obj.scaledown.w( data.w );
                end
            else
                data_scaled = struct;    % initialize
                data_scaled.t = data.t;  % time is not scaled
                data_scaled.y = obj.scaleup.y( data.y );    %data.y * obj.params.scaleup.y;
                data_scaled.u = obj.scaleup.u( data.u );    %data.u * obj.params.scaleup.u;
                if ismember( 'x' , fields(data) )
                    data_scaled.x = data.scaleup.x( data.x );    %data.x * obj.params.scaleup.x;
                end
                if ismember( 'w' , fields(data) )
                    data_scaled.w = data.scaleup.w( data.w );    %data.x * obj.params.scaleup.x;
                end
            end
        end
        
        % chop (chop data into several trials)
        function data_chopped = chop( obj , data , num , len )
            %chop: chop data into num trials of lenght len
            %   data - struct with fields t , y , (x)
            %   data_chopped - cell array containing the chopped datas
            
            % determine length of timestep
            Ts = mean( data.t(2:end) - data.t(1:end-1) ); % take mean in case they're not quite uniform
            
            % find maximum length of each chop for given num
            maxlen = data.t(end) / num;
            if len > maxlen
                len = maxlen;
                disp([ 'Maximum trial length is ' , num2str(maxlen) , 's. Using this value instead.' ]);
            end
            
            % set length of the chops in terms of time steps
            lenk = length( find( data.t < len ) );
            maxlenk = length( find( data.t < maxlen ) );
            
            data_chopped = cell( 1 , num );
            for i = 1 : num
                index = (i-1) * maxlenk + ( 1 : lenk );
                
                % chop the data
                data_chopped{i}.t = ( ( 1 : lenk ) - 1 ) * Ts;
                data_chopped{i}.y = data.y( index , : );
                data_chopped{i}.u = data.u( index , : );
                if ismember( 'x' , fields(data) )
                    data_chopped{i}.x = data.x( index , : );
                end
            end  
        end
        
        % merge_trials (merge cell array containing several sim/exp trials into one data struct)
        function data_merged = merge_trials( obj , data )
            %merge_trials: Merge cell array containing several sim/exp trials into one data struct
            %   data - cell array where each cell is a data struct with
            %   fields t, y, u, (x), (params), ...
            %   data_merged: data struct with the same fields
            
            % confirm that data is a cell array (i.e. contains several trials)
            % If so, concatenate all trials into a single data struct 
            if iscell( data )
                data_merged = data{1};  % initialize the merged data struct
                for i = 2 : length( data )
                    data_fields = fields( data{i} );
                    for j = 1 : length( data_fields )
                        if isa( data{i}.( data_fields{j} ) , 'numeric' )
                            data_merged.( data_fields{j} ) = [ data_merged.( data_fields{j} ) ; data{i}.( data_fields{j} ) ];
                        end
                    end
                end
            else
                data_merged = data; % if not a cell array, do nothing
            end
        end
        
        %% save the class object
        
        % save_class
        function obj = save_class( obj , model_id )
            %save_class: Saves the class as a .mat file
            %   If class is from a simulated system, it is saved in the
            %   subfolder corresponding to that system.
            %   If class if from a real system, it is saved in the generic
            %   folder /systems/fromData/.
            %   varargin = isupdate - indicates whether this is an update of a
            %   previously saved class (1) or a new class (0).
            
            % shorthand
            isupdate = obj.isupdate;
            
            % if no model id is provided, save the first candidate model
            if nargin < 2 && ~iscell(obj.candidates)
                obj.model = obj.candidates;
            elseif nargin < 2
                obj.model = obj.candidates{1};
            else
                obj.model = obj.candidates{model_id};
            end
                        
            % set the class name based on its parameters
            if isupdate
                classname = obj.params.classname;
            else
                dateString = datestr(now , 'yyyy-mm-dd_HH-MM'); % current date/time
                classname = [ obj.model_type , '_' , obj.obs_type{1} , '-' , num2str(obj.obs_degree) ,'_n-' , num2str( obj.params.n ) , '_m-' , num2str( obj.params.m ) , '_del-' , num2str( obj.params.nd ) , '_' , dateString ];
                obj.params.classname = classname;   % create classname parameter
            end
    
            % save the class file
            sysid_class = obj;
            if obj.params.isfake    % check if class was created from simulated system
                dirname = [ 'systems' , filesep , obj.params.sysParams.sysName , filesep , 'models'];
                if ~isupdate
                    mkdir( dirname );
                end
                fname = [ dirname , filesep , classname, '.mat' ];
                save( fname , 'sysid_class' );
            else
                dirname = [ 'systems' , filesep , 'fromData' ];
                fname = [ dirname , filesep , classname, '.mat' ];
                save( fname , 'sysid_class' , '-v7.3' );
            end
        end
        
        %% defining observables
        
        % def_observables (define the basis of observable functions)
        function obj = def_observables( obj , type , degree )
            % def_observables: Defines the set of nonlinear observable
            % functions that will act as basis of Koopman subspace
            %   type - Type of functions, [cell array of strings].
            %       'armshape' - coefficients of shape polynomial
            %       'poly'  - monomials
            %       ... more to be added over time
            %   degree - Maximum degree/complexity of functions, [vector].
            
            % check that the type and degree have same dimension
            if size(type) ~= size(degree)
                error('inputs must be of the same size');
            end
            
            % define the low dimensional state with delays, called zeta
            x = sym('x', [obj.params.n, 1] , 'real');   % state variable x
            xd = sym('xd', [obj.params.nd * obj.params.n, 1] , 'real');   % state delays i.e. for 2 delays: [x_i-1, x_i-2]'
            ud = sym('ud', [obj.params.nd * obj.params.m, 1] , 'real');   % input delays i.e. for 2 delays: [u_i-1, u_i-2]'
            zeta = [x ; xd; ud];    % state variable with delays
            u = sym('u', [obj.params.m, 1] , 'real');   % input vector u
            if strcmp( obj.model_type , 'nonlinear' )    % if the includes input in unlifted state
                zeta = [ zeta ; u ];    % just so lifting function works
            end
            obj.params.zeta = zeta; % needed for defining lifting function
            obj.params.x = x;
            obj.params.u = u;
            
            disp('Defining initial set of observables...');
            % construct the observables
            fullBasis = zeta;  % first nzeta observables should always be the measured state and delays
            for i = 1 : length(degree)
                if strcmp( type{i} , 'poly' )
                    obj = obj.def_polyLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.poly( size(zeta,1)+1 : end ) ]; % don't include first nzeta states because they will be repeats
                elseif strcmp( type{i} , 'fourier' )
                    obj = obj.def_fourierLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.fourier ];
                elseif strcmp( type{i} , 'fourier_sparser' )
                    obj = obj.def_fourierLift_sparser( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.fourier_sparser ];
                elseif strcmp( type{i} , 'gaussian' )
                    obj = obj.def_gaussianLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.gaussian ];
                elseif strcmp( type{i} , 'hermite' )
                    obj = obj.def_hermiteLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.hermite ];
                end
            end
            
            % add a constant term to the end of the set
            fullBasis = [ fullBasis ; sym(1) ];
            
            % BILINEAR LIFT: multiply input by monomials of state, but no powers of input
            if strcmp( obj.model_type , 'bilinear' )   
                % modify basis to include inputs
                basis_diag = kron( eye(obj.params.m) , fullBasis );
                fullBasis_input = [ fullBasis ; basis_diag * u ];
                
                % save basis and function to class
                obj.basis.full_input = fullBasis_input; % symbolic expression
                obj.lift.full_input = matlabFunction( fullBasis_input , 'Vars' , { zeta , u } );  % lifting function
            end
            
            % remove current input from zeta
            if strcmp( obj.model_type , 'nonlinear' )
                zeta = zeta(1 : obj.params.nzeta);
            end
            
            % define outputs
            obj.params.zeta = zeta;
            obj.basis.full = fullBasis;
            obj.lift.full = matlabFunction( fullBasis , 'Vars' , { [ zeta ; u ] } );
            obj.params.N = length( fullBasis ); % the dimension of the lifted state

        end
        
        % def_observables_loaded (define the basis of observable functions)
        function obj = def_observables_loaded( obj , type , degree )
            % def_observables_loaded: Defines the set of nonlinear observable
            % functions that will act as basis of Koopman subspace, when
            % the model includes loads.
            %   type - Type of functions, [cell array of strings].
            %       'armshape' - coefficients of shape polynomial
            %       'poly'  - monomials
            %       ... more to be added over time
            %   degree - Maximum degree/complexity of functions, [vector].
            
            % check that the type and degree have same dimension
            if size(type) ~= size(degree)
                error('inputs must be of the same size');
            end
            
            % define the low dimensional state with delays, called zeta
            x = sym('x', [obj.params.n, 1] , 'real');   % state variable x
            xd = sym('xd', [obj.params.nd * obj.params.n, 1] , 'real');   % state delays i.e. for 2 delays: [x_i-1, x_i-2]'
            ud = sym('ud', [obj.params.nd * obj.params.m, 1] , 'real');   % input delays i.e. for 2 delays: [u_i-1, u_i-2]'
            zeta = [x ; xd; ud];    % state variable with delays
            u = sym('u', [obj.params.m, 1] , 'real');   % input vector u
            w = sym('w', [obj.params.nw, 1] , 'real');  % load vector w
            if strcmp( obj.model_type , 'nonlinear' )    % if the includes input in unlifted state
                zeta = [ zeta ; u ];    % just so lifting function works
            end
            obj.params.zeta = zeta; % needed for defining lifting function
            obj.params.x = x;
            obj.params.u = u;
            obj.params.w = w;
            
            disp('Defining initial set of observables...');
            % construct the observables
            fullBasis = zeta;  % first nzeta observables should always be the measured state and delays
            for i = 1 : length(degree)
                if strcmp( type{i} , 'poly' )
                    obj = obj.def_polyLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.poly( size(zeta,1)+1 : end ) ]; % don't include first nzeta states because they will be repeats
                elseif strcmp( type{i} , 'fourier' )
                    obj = obj.def_fourierLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.fourier ];
                elseif strcmp( type{i} , 'fourier_sparser' )
                    obj = obj.def_fourierLift_sparser( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.fourier_sparser ];
                elseif strcmp( type{i} , 'gaussian' )
                    obj = obj.def_gaussianLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.gaussian ];
                elseif strcmp( type{i} , 'hermite' )
                    obj = obj.def_hermiteLift( degree(i) );
                    fullBasis = [ fullBasis ; obj.basis.hermite ];
                end
            end
            
            % add a constant term to the end of the set
            fullBasis = [ fullBasis ; sym(1) ];
            
            % incorporate the loads into the basis set
            zw = [ 1 ; w ];    % load vector, with 1 appended to beginning
            fullBasis_loaded = fullBasis;% different version, doesn't requare large symbolic matrices
            for i = 1 : obj.params.nw
                fullBasis_loaded = [ fullBasis_loaded ; w(i) * fullBasis ];
            end
            
            % BILINEAR LIFT: multiply input by monomials of state, but no powers of input
            if strcmp( obj.model_type , 'bilinear' )        
                % modify basis to include inputs
                basis_diag = kron( eye(obj.params.m) , fullBasis_loaded );
                fullBasis_loaded_input = [ fullBasis_loaded ; basis_diag * u ]; % save to class
                
                % save basis and function to class
                obj.basis.full_loaded_input = fullBasis_loaded_input; % symbolic expression
                obj.lift.full_loaded_input = matlabFunction( fullBasis_loaded_input , 'Vars' , { zeta , w , u} );  % lifting function
            end
            
            % remove current input from zeta
            if strcmp( obj.model_type , 'nonlinear' )
                zeta = zeta(1 : obj.params.nzeta);
            end
            
            % define outputs
            obj.params.zeta = zeta;
            obj.params.zw = zw;
            obj.basis.full = fullBasis;
            obj.basis.full_loaded = fullBasis_loaded;
            obj.lift.full = matlabFunction( fullBasis , 'Vars' , { [ zeta ; u ] } );
            obj.lift.full_loaded = matlabFunction( fullBasis_loaded , 'Vars' , { [ zeta ; u ] , w } );
            obj.params.N = length( fullBasis ); % the dimension of the lifted state

        end
        
        % def_polyLift (defines polynomial basis functions)
        function [ obj , polyBasis ] = def_polyLift( obj , degree )
            %def_polyLift: Defines the lifting function that lifts state variable x to
            % space spanned by monomials with total degree less than or equal to
            % max_degree.
            %   e.g. 1 x1 x2 x1^2 x1x2 x2^2 ...
            
            zeta = obj.params.zeta; % get the symbolic unlifted state
            nzeta = length(zeta);
            maxDegree = degree;
            
            % Number of mononials, i.e. dimenstion of p(x)
%             N = factorial(nzeta + maxDegree) / ( factorial(nzeta) * factorial(maxDegree) );
            N = prod( ( nzeta + 1) : ( nzeta + maxDegree ) ) / factorial(maxDegree);    % avoids Infs for really big factorials

            
            % matrix of exponents (N x naug). Each row gives exponents for 1 monomial
            exponents = [];
            for i = 1:maxDegree
                exponents = [exponents; partitions(i, ones(1,nzeta))];
            end
%             exponents = [exponents ; zeros(1,nzeta)];   % put constant at end of basis so state can be the first nzeta elements
            
            % create vector of orderd monomials (column vector)
            for i = 1:N-1
                polyBasis(i,1) = obj.get_monomial(zeta, exponents(i,:));
            end
            
%             % define matrix of exponents: columns=monomial term, rows=dimension of x
%             psi = exponents';
            
            % create the lifting function: zeta -> p(zeta)
            obj.lift.poly = matlabFunction(polyBasis, 'Vars', {zeta});
%             
%             % define derivative of lifted state with respect to x
%             dlift = jacobian(polyBasis,x);
%             matlabFunction(dlift, 'File', 'jacobianLift', 'Vars', {zeta});
            
            % output variables
            obj.basis.poly = polyBasis;    % symbolic vector of basis monomials, p(x)
%             ams.jacobianBasis = dlift;
%             params.N = N;   % dimension of polyBasis (including the state itself)
%             params.Np = N + p; % dimension of the lifted state
%             params.psi = psi;   % monomial exponent index function
%             params.x = x;   % symbolic state variable
%             params.u = u;   % symbolic input variable
%             params.xd = xd; % symbolic state delays
%             params.ud = ud; % symbolic input delays
            
        end
        
        % get_monomial (builds a monomial)
        function [ monomial ] = get_monomial( obj, x, exponents )
            %get_monomial: builds a monomial from symbolic vector x and a vector of
            %exponents
            %   e.g. x = [x1 x2]; exponents = [1 2]; =>  monomial = x1^1 * x2^2
            
            n = length(x);
            
            monomial = x(1)^exponents(1);
            for i = 2:n
                monomial = monomial * x(i)^exponents(i);
            end
        end
        
        % def_fourierLift (defines sin/cos basis functions)
        function [ obj , fourierBasis ] = def_fourierLift( obj , degree )
            %def_fourierLift: Defines fourier basis functions
            
            % shorthand variable names
            n = obj.params.n;
            p = obj.params.m;
            zeta = obj.params.zeta; % get the symbolic unlifted state
            nzeta = length(zeta);
            maxDegree = degree;
            
            % Number of basis elements, i.e. dimenstion of p(x)
            Nfourier = nzeta + (1 + 2*maxDegree)^nzeta;
            
            % create sines of cosines of all the states
            poop = sym( zeros(1+2*maxDegree , nzeta) );
            for i = 1 : nzeta
                poop(1,i) = 1;
                for j = 1 : maxDegree
                    poop(2*j,i)   = cos(2*pi*j*zeta(i));
                    poop(2*j+1,i) = sin(2*pi*j*zeta(i));
                end
            end
            
            % define fourier basis vector
            fourierBasis = poop(:,1);
            for i = 2 : nzeta
                fourierBasis = kron(fourierBasis, poop(:,i));
            end
            
            % remove the constant element from the basis
            fourierBasis = fourierBasis( 2 : end );
            
            % create the lifting function: zeta -> fourier(zeta)
            obj.lift.fourier = matlabFunction(fourierBasis, 'Vars', {zeta});
            
            % output variables
            obj.basis.fourier = fourierBasis;    % symbolic vector         
        end
        
        % def_fourierLift_sparser (defines sin/cos basis functions)
        function [ obj , fourierBasis ] = def_fourierLift_sparser( obj , degree )
            %def_fourierLift_sparser: Defines fourier basis functions, but
            % not as many as def_fourierLift (i.e. not every combination of
            % a suitable degree)
        
            % shorthand variable names
            n = obj.params.n;
            p = obj.params.m;
            zeta = obj.params.zeta; % get the symbolic unlifted state
            nzeta = length(zeta);
            maxDegree = degree;
            
            % matrix of exponents (N x naug). Each row gives exponents for 1 monomial
            multipliers = zeros(1,2*nzeta);
            for i = 1:maxDegree
                multipliers = [multipliers; partitions(i, ones(1, 2*nzeta))];
            end
            multipliers = multipliers( 2 : end , : );   % remove 1st row which is a constant
            
            % Number of basis elements, i.e. dimenstion of p(x)
            N = nzeta + size(multipliers , 1);
            
            % create vector of sines and cosines with multipliers
            fourierBasis = sym('fourierBasis', [N-nzeta,1]);
            for i = 1:N-nzeta
                fourierBasis(i,1) = obj.get_sinusoid(zeta, multipliers(i,:));
            end
            
            % create the lifting function: zeta -> fourier_sparser(zeta)
            obj.lift.fourier_sparser = matlabFunction(fourierBasis, 'Vars', {zeta});
            
            % output variables
            obj.basis.fourier_sparser = fourierBasis;    % symbolic vector
        end
        
        % get_sinusoid (builds a sinusoid from symbolic vector x and a vector of multipliers)
        function [ sinusoid ] = get_sinusoid( obj , x , multiplier )
            %get_sinusoid: builds a sinusoid from symbolic vector x and a vector of multipliers
            %   e.g. x = []
            
            n = length(multiplier); % vector of multipliers
            
            sinusoid = sym(1);  % initialize as a symbolic variable
            for i = 1 : n/2
                if multiplier(i) ~= 0
                    sinusoid = sinusoid * sin(2*pi*multiplier(i)*x(i));
                end
            end
            for j = n/2 + 1 : n
                if multiplier(j) ~= 0
                    sinusoid = sinusoid * cos(2*pi*multiplier(j)*x(j - n/2));
                end
            end
        end
                   
        % def_gaussianLift (defines gaussian basis functions)
        function [ obj , gaussianBasis ] = def_gaussianLift( obj , degree )
            %def_gaussianLift: Defines a set of randomly distributed
            %gaussian basis functions
            
            % shorthand variable names
            n = obj.params.n;
            p = obj.params.m;
            zeta = obj.params.zeta; % get the symbolic unlifted state
            nzeta = length(zeta);
            maxDegree = degree;
            
            % create basis functions with random centers in interval [-1 , 1]
            psi = sym('gaussianBasis', [maxDegree , 1]);
            zeta0 = (2*rand([nzeta,maxDegree]) - 1); % columns are random centers
            for i = 1 : maxDegree
                radius = norm( zeta - zeta0(:,i) );
                gaussianBasis(i,:) = exp(-( 1 * radius )^2) ;
                %    % I think this might work faster
                %    radius = sum( ( zeta - zeta0(:,i) ).^2 );
                %    psi(i,:) = exp( -radius );
            end
            
            % create the lifting function: zeta -> fourier_sparser(zeta)
            obj.lift.gaussian = matlabFunction(gaussianBasis, 'Vars', {zeta});
            
            % output variables
            obj.basis.gaussian = gaussianBasis;    % symbolic vector
        end
        
        % get_hermite (builds a product of hermite polynomials)
        function [ hermite ] = get_hermite( obj, x, orders )
            %get_monomial: builds a monomial from symbolic vector x and a vector of
            %exponents
            %   e.g. x = [x1 x2]; exponents = [1 2]; =>  monomial = hermiteH(1,x1) * hermiteH(2,x2)
            
            n = length(x);
            
            hermite = hermiteH( orders(1) , x(1) );
            for i = 2:n
                hermite = hermite * hermiteH( orders(i) , x(i) );
            end
        end
        
        % def_polyLift (defines polynomial basis functions)
        function [ obj , hermiteBasis ] = def_hermiteLift( obj , degree )
            %def_polyLift: Defines the lifting function that lifts state variable x to
            % space spanned by monomials with total degree less than or equal to
            % max_degree.
            %   e.g. 1 x1 x2 x1^2 x1x2 x2^2 ...
            
            zeta = obj.params.zeta; % get the symbolic unlifted state
            nzeta = length(zeta);
            maxDegree = degree;
            
            % Number of mononials, i.e. dimenstion of p(x)
            N = factorial(nzeta + maxDegree) / ( factorial(nzeta) * factorial(maxDegree) );
            
            % matrix of exponents (N x naug). Each row gives exponents for 1 monomial
            exponents = [];
            for i = 1:maxDegree
                exponents = [exponents; partitions(i, ones(1,nzeta))];
            end
            
            % create vector of orderd monomials (column vector)
            for i = 1:N-1
                hermiteBasis(i,1) = obj.get_hermite(zeta, exponents(i,:));
            end
            
            % create the lifting function: zeta -> p(zeta)
            obj.lift.hermite = matlabFunction(hermiteBasis, 'Vars', {zeta});
            
            % output variables
            obj.basis.hermite = hermiteBasis;    % symbolic vector of basis monomials, p(x)
        end
        
        %% fitting Koopman operator and A,B,C system matrices
        
        % get_zeta (adds a zeta field to a test data struct)
        function [ data_out , zeta ] = get_zeta( obj , data_in )
            %get_zeta: Adds a zeta field to a test data struct
            %   data_in - struct with t , x , y , u fields
            %   zeta - [ y , yd1 , yd2 , ... , ud1 , ud2 , ... ]
            
            data_out = data_in;
            
            if obj.delays ~= 0
                % add the zeta field
                for i = obj.params.nd + 1 : size( data_in.y , 1 )
                    ind = i - obj.params.nd;    % current timestep index
                    y = data_in.y( i , : );
                    u = data_in.u( i , : );
                    ydel = zeros( 1 , obj.params.nd * obj.params.n );
                    udel = zeros( 1 , obj.params.nd * obj.params.m );
                    for j = 1 : obj.params.nd
                        fillrange_y = obj.params.n * (j - 1) + 1 : obj.params.n * j;
                        fillrange_u = obj.params.m * (j - 1) + 1 : obj.params.m * j;
                        ydel(1 , fillrange_y) = data_in.y( i - j , : );
                        udel(1 , fillrange_u) = data_in.u( i - j , : );
                    end
                    zetak = [ y , ydel , udel ];
                    %                 if strcmp( obj.model_type , 'nonlinear' )     % include input in zeta
                    %                     zetak = [ zetak , u ];
                    %                 end
                    data_out.zeta( ind , : ) = zetak;
                    data_out.uzeta( ind , : ) = data_in.u( i , : );    % current timestep with zeta (input starting at current timestep)
                    if isfield( data_in , 'w' )
                        data_out.wzeta( ind , : ) = data_in.w( i , : );
                    end
                end
            else
                data_out.zeta = data_in.y;  % if no delays, zeta is equal to y
                data_out.uzeta = data_in.u;
                if isfield( data_in , 'w' )
                    data_out.wzeta = data_in.w;
                end
            end
            zeta = data_out.zeta;
        end
        
        % get_snapshotPairs (convert time-series data into snapshot pairs)
        function snapshotPairs = get_snapshotPairs( obj , data , varargin )
            %get_snapshotPairs: Convert time-series data into a set of num
            %snapshot pairs.
            %   data - struct with fields x , y , u , t , (zeta) OR cell
            %     array containing cells which contain those fields
            %   varargin = num - number of snapshot pairs to be taken
            
            disp('Constructing snapshots...');
            
            % check wheter data is a cell array (i.e. contains several trials)
            % If so, concatenate all trials into a single data struct 
            if iscell( data )
                data_merged = obj.merge_trials( data );
                data = data_merged; % replace cell array with merged data struct
            end
            
            % check if data has a zeta field, create one if not
            if ~ismember( 'zeta' , fields(data) )
                data = obj.get_zeta( data );
            end
            
            % If snapshots are already defined in data file use those
            if ismember( 'snapshots' , fields(data) )
                snapshotPairs.alpha = data.snapshots.alpha;
                snapshotPairs.beta = data.snapshots.beta;
                snapshotPairs.u = data.snapshots.u;
                if ismember( 'w' , fields(data.snapshots) )
                    snapshotPairs.w = data.snapshots.w;
                end
            else    % otherwise, construct them from time-series data 
                % separate data into 'before' and 'after' time step
                before.t = data.t( obj.params.nd + 1 : end-1 );
                before.zeta = data.zeta( 1:end-1 , : );
                after.t = data.t( obj.params.nd + 2 : end );
                after.zeta = data.zeta( 2:end , : );
                u = data.uzeta( 1:end-1 , : );    % input that happens between before.zeta and after.zeta
                
                % remove pairs that fall at the boundary between sim/exp trials
                goodpts = find( before.t < after.t );
                before.zeta = before.zeta( goodpts , : );
                after.zeta = after.zeta( goodpts , : );
                u = u( goodpts , : );
                
                % if system is loaded, include the load
                if isfield( data , 'w' )
                    w = data.wzeta( 1:end-1 , : );    % load that happens between before.zeta and after.zeta
                    w = w( goodpts , : );
                end
                
                % set the number of snapshot pairs to be taken
                num_max = size( before.zeta , 1 ) - 1; % maximum number of snapshot pairs
                if length(varargin) == 1
                    num = varargin{1};
                    if num > num_max - 1
                        message = [ 'Number of snapshot pairs cannot exceed ' , num2str(num_max) , '. Taking ' , num2str(num_max) , ' pairs instead.' ];
                        disp(message);
                        num = num_max;
                    end
                else
                    num = num_max;
                end
                
                % randomly select num snapshot pairs
                total = num_max;
                s = RandStream('mlfg6331_64');
                index = datasample(s , 1:total, num , 'Replace' , false);
                
                snapshotPairs.alpha = before.zeta( index , : );
                snapshotPairs.beta = after.zeta( index , : );
                snapshotPairs.u = u( index , : );
                if isfield( data , 'w' )
                    snapshotPairs.w = w( index , : );
                end
            end
        end
        
        % get_Koopman (Find the best possible koopman operator from snapshot pairs)
        function [ koopData , K ] = get_Koopman( obj ,  snapshotPairs , varargin )
            %get_KoopmanConstGen: Find the best possible koopman operator given
            %snapshot pairs using constraint generation to deal with large data sets.
            %   varargin = lasso weighting parameter. lasso >> 1 approaches least squares solution 
            
            if length(varargin) == 1
                if isempty( varargin{1} )
                    lasso = 1e4 * obj.params.N;   % defualt value of the lasso parameter (should emulate least squares)
                else
                    lasso = varargin{1} * obj.params.N;
                end
            else
                lasso = 1e4 * obj.params.N;   % defualt value of the lasso parameter (should emulate least squares)
            end
            
            disp('Finding Koopman operator approximation...');
            
            % Extract snapshot pairs
            [x,y,u] = deal( snapshotPairs.alpha , snapshotPairs.beta , snapshotPairs.u );
            if isfield( snapshotPairs , 'w' )
                w = snapshotPairs.w;
                nw = obj.params.nw;
            else
                nw = 0; % null nw for when there is no load
            end
            
            % Build matrices
            [~, m] = deal( obj.params.n , obj.params.m );
            Nm = obj.params.N + m;   % dimension of z plus input
            N = obj.params.N;       % dimension of z (i.e. lifted state)
            
            % preallocation
            if strcmp( obj.model_type , 'nonlinear' )    % don't append input
                Px = zeros(length(x), N * (nw+1) );
                Py = zeros(length(x), N * (nw+1) );
            elseif strcmp( obj.model_type , 'bilinear' )    % don't append input
                Px = zeros(length(x), ( N * (nw+1) ) * (m + 1) );
                Py = zeros(length(x), ( N * (nw+1) ) * (m + 1) );
            else    % append input to end of lifted state
                Px = zeros( length(x), N * (nw+1) + m );
                Py = zeros( length(x), N * (nw+1) + m );
            end
            disp('Evaluating basis functions on snapshots...');
            for i = 1:length(x)
                if ~mod( i , floor(length(x)/10) )  % only update progress once in a while
                    obj.loop_progress( i , length(x) );
                end
                if strcmp( obj.model_type , 'nonlinear' )    % don't append input if it already is lifted nonlinearly
                    if obj.loaded
                        psix = obj.lift.econ_full_loaded( [ x(i,:) , u(i,:) ]' , w(i,:)' )';   
                        psiy = obj.lift.econ_full_loaded( [ y(i,:) , u(i,:) ]' , w(i,:)' )'; 
                    else
                        psix = obj.lift.econ_full( [ x(i,:) , u(i,:) ]' )';   
                        psiy = obj.lift.econ_full( [ y(i,:) , u(i,:) ]' )';    
                    end
                    Px(i,:) = psix;
                    Py(i,:) = psiy;
                elseif strcmp( obj.model_type , 'bilinear' )    % don't append input if it already is lifted
                    if obj.loaded
                        psix = obj.lift.econ_full_loaded_input( x(i,:)' , w(i,:)' , u(i,:)' )';   
                        psiy = obj.lift.econ_full_loaded_input( y(i,:)' , w(i,:)' , u(i,:)' )'; 
                    else
                        psix = obj.lift.econ_full_input( x(i,:)' , u(i,:)' )';   
                        psiy = obj.lift.econ_full_input( y(i,:)' , u(i,:)' )';    
                    end
                    Px(i,:) = psix;
                    Py(i,:) = psiy;
                else
                    if obj.loaded
                        psix = obj.lift.econ_full_loaded( x(i,:)' , w(i,:)' )';
                        psiy = obj.lift.econ_full_loaded( y(i,:)' , w(i,:)' )';
                    else
                        psix = obj.lift.econ_full( x(i,:)' )';   
                        psiy = obj.lift.econ_full( y(i,:)' )';
                    end
                    Px(i,:) = [ psix , u(i,:) ];
                    Py(i,:) = [ psiy , u(i,:) ];     % uses same u as Px
                end
            end
            
            % Call function that solves QP problem
            if obj.lasso >= 1e6     % least-squares
                K = Px \ Py;    % least-squares solution (very efficient, but no L1 penalty)
            else    % LASSO 
                Uvec = obj.solve_KoopmanQP( Px , Py , lasso);
                if strcmp( obj.model_type , 'nonlinear' )
                    Umtx = reshape(Uvec, [ N*(nw+1) , N*(nw+1) ]); % Koopman operator matrix
                elseif strcmp( obj.model_type , 'bilinear' )
                    Umtx = reshape(Uvec, [ N*(nw+1)*(m+1) , N*(nw+1)*(m+1) ]); % Koopman operator matrix
                else
                    Umtx = reshape(Uvec, [ N*(nw+1)+m , N*(nw+1)+m ]); % Koopman operator matrix
                end
                K = Umtx;   % switching to K convention to not confuse with input
            end
            
            
            % other usefule outputs
            koopData.K = K; % Koopman operator matrix (note the switch to K)
            koopData.Px = Px( : , 1 : N*(nw+1) );   % only want state, not input
            koopData.Py = Py( : , 1 : N*(nw+1) );
            koopData.u = u; % input
            if isfield( snapshotPairs , 'w' )
                koopData.w = w; % load(s)
            end
            koopData.alpha = snapshotPairs.alpha;
        end
        
        % solve_KoopmanQP (Finds the Koopman operator using Lasso regression)
        function Uvec = solve_KoopmanQP( obj , Px , Py , lasso )
            %solve_KoopmanQP: Finds the Koopman operator for a given set of
            %data using the lasso regression method.
            %   Lasso method
            %   x is vectorized Koopman operator, decomposed into positive and negative parts of each entry x = [u11+, ..., uNN+, u11-, ... , uNN-]';
            %   Uvec = M * x, where M subtracts the + and - parts of each entry: uij+ - uij-
            
            % local variable names
            nx = obj.params.N^2;
            Nm = size( Px , 2 );    % obj.params.N + obj.params.m;
            N = obj.params.N;
            n = obj.params.n;
            m = obj.params.m;
            nd = obj.params.nd;
            nnd = obj.params.n * obj.params.nd;
            mnd = obj.params.m * obj.params.nd;
            
            M = [speye(Nm^2) , -speye(Nm^2)];
            
            PxTPx = Px' * Px;
            
            % Ensure PxTPx is PSD by adding constant along diagonal
            [ ~ , Dpx ] = eig(PxTPx);
            if any( diag(Dpx) < 0 )
                PxTPx = PxTPx + eye( size(PxTPx,1) ) * 1e-6;
            end
%             [ U,S,V ] = svd(PxTPx);
%             S( find(abs(S) < 1e-10) ) = 0;
%             PxTPx = U * S * V';

            PxTPy = Px' * Py;
            ATA = kron(speye(Nm) , PxTPx);  % repeat blocks diagonally N times
            ATb = reshape(PxTPy, [Nm^2 , 1]);
            
            % L2 error as cost function
            preH = ATA * M;
            H = M' * preH;
            f = -M' * ATb;
            
            % L1 regularization enforced as constraint
            t = lasso;
            Aq = [ -speye(2*Nm^2) ; ones(1 , 2*Nm^2) ];
            bq = [ zeros(2*Nm^2 , 1) ; t ];
            
            if obj.liftinput == 0   % only do this for linear models
                % enforce delay constraint (see notebook from 2019-8-22)
                if nd >= 1
                    Ad_pos = speye( Nm^2 );
                    Ad_pos = Ad_pos( n*Nm+1 : Nm*( n*(nd+1) + mnd ) , : );  % remove unused rows
                    bd_pos = zeros( Nm * ( nnd + mnd ) , 1 );
                    for i = 1 : nnd     % state delays
                        index = (Nm+1) * (i-1) + 1;
                        bd_pos(index,1) = 1;
                    end
                    for i = 1 : m   % first input delay
                        index = Nm * nnd + N + (Nm+1) * (i-1) + 1;
                        bd_pos(index,1) = 1;
                    end
                    for i = 1 : m*(nd-1)    % subsequent input delays (nd > 1)
                        index = Nm * (nnd+m) + nnd + (Nm+1) * (i-1) + 1;
                        bd_pos(index,1) = 1;
                    end
                    Ad = [ Ad_pos ; -Ad_pos ] * M;
                    bd = [ bd_pos ; -bd_pos ];
                    
                    % tack on the the delay constraint
                    Aq = [ Aq ; Ad ];
                    bq = [ bq ; bd ];
                end
            end
            
            % Solve the quadratic program
%             [x , results] = quadprog_gurobi( 0.5*H , f , Aq , bq );       % use gurobi to solve
            options = optimoptions('quadprog', 'Display', 'iter');
            [ x, fval, exitflag ] = quadprog(H, f, Aq, bq, [], [], [], [], [],options);      % use matlab to solve
            
            % Recover Uvec from the optimization variable
            xout = M * x;
            
            % Set output
            Uvec = xout;
        end
        
        % get_model (Extracts the A,B,C matrices from Koopman matrix)
        function [ out , obj ] = get_model( obj , koopData )
            %get_model: Defines the A, B, and C matrices that describe the linear
            %lifted system z+ = Az + Bu, x = Cz.
            %   out - struct with fields A, B, C, sys, params, ...
            %   obj.model - property of struct which stores the model
            
            % convert to continuous time model
            if strcmp( obj.time_type , 'continuous' )
                UT = (1/obj.params.Ts) * logm( koopData.K' + 1e-12 * eye(size(koopData.K)) );    % apply conditioning
            else
                UT = koopData.K';    % transpose of koopman operator
            end
            
            if isfield( koopData , 'w' )
                nw = obj.params.nw;
            else
                nw = 0; % null nw for when there is no load
            end
            
            % Extract the A and B system matrices
            A = UT( 1 : obj.params.N * (nw+1) , 1 : obj.params.N * (nw+1) );
            B = UT( 1 : obj.params.N * (nw+1) , obj.params.N * (nw+1) + 1 : end );
            
            % Cy selects the first n entries of the lifted state
            Cy = [eye(obj.params.n), zeros(obj.params.n , obj.params.N*(nw+1) - obj.params.n)];
            
            % find M matrix that (approximately) projects a lifted point onto the subset of all legitimate lifted points in R^N
            Px = koopData.Px; Py = koopData.Py;
            U = koopData.u;
            L = zeros( size(Px,1) , obj.params.N * (nw+1) );
            for i = 1 : size( Px , 1)
                L(i,:) = ( A * Px(i,:)' + B * U(i,:)' )' ;        % with input
            end
            R = zeros( size(L,1) , obj.params.N * (nw+1));
            for i = 1 : size( L , 1 )
                R(i,:) = Py(i,:) ;
            end
            Mtranspose = L \ R;
            M = Mtranspose';
            
            % define outputs
            if strcmp( obj.time_type , 'continuous' )
                out.A = A;  % edited to include projection M, 12/11/2018
                out.B = B;  % edited to include projection M, 12/11/2018
            else
                out.A = M*A;  % edited to include projection M, 12/11/2018
                out.B = M*B;  % edited to include projection M, 12/11/2018
            end
            out.C = Cy;
            out.M = M;
            out.sys = ss( out.A , out.B , Cy , 0 , obj.params.Ts );  % discrete state space system object
            out.params = obj.params;    % save system parameters as part of system struct    
            out.K = koopData.K; % save the Koopman operator matrix just in case
            
            % add model substruct to the class
            obj.model = out;
        end
        
        % get_BLmodel (Extracts bilinear model from the Koopman matrix)
        function [ out , obj ] = get_BLmodel( obj , koopData )
            %get_BLmodel: Defines the A, B, Beta, and C matrices that describe the linear
            %lifted system z+ = Az + Beta(z)u, y = Cz.
            %   out - struct with fields A, B, C, sys, params, ...
            %   obj.model - property of struct which stores the model
            
            % convert to continuous time model
            if strcmp( obj.time_type , 'continuous' )
                UT = (1/obj.params.Ts) * logm( koopData.K' + 1e-12 * eye(size(koopData.K)) );    % apply conditioning
            else
                UT = koopData.K';    % transpose of koopman operator
            end
            
            if isfield( koopData , 'w' )
                nw = obj.params.nw;
            else
                nw = 0; % null nw for when there is no load
            end
            
            % Extract the A and B system matrices
            A = UT( 1 : obj.params.N * (nw+1) , 1 : obj.params.N * (nw+1) );
            B = UT( 1 : obj.params.N * (nw+1) , obj.params.N * (nw+1) + 1 : end );
            
            % Cy selects the first n entries of the lifted state
            Cy = [eye(obj.params.n), zeros(obj.params.n , obj.params.N*(nw+1) - obj.params.n)];
            
            % bilinear term matrix (depends on zeta)
            Beta = @(z_in)obj.get_Beta_bilinear( B , z_in );
%             if obj.loaded
%                 Beta = @(zeta_in,w_in)obj.get_Beta_bilinear(B,zeta_in,w_in);
%             else
%                 Beta = @(zeta_in)obj.get_Beta_bilinear(B,zeta_in);
%             end
                
            % define outputs
            out.A = A;
            out.B = B;
            out.Beta = Beta;   % bilinear term matrix function
            out.C = Cy;
            out.params = obj.params;    % save system parameters as part of system struct    
            out.K = koopData.K; % save the Koopman operator matrix just in case
            
            % add model substruct to the class
            obj.model = out;
        end
        
        % get_Beta_bilinear: Constructs the model state dependent B matrix
        function Beta = get_Beta_bilinear( obj , B , z )
            if nargin < 4
%                 zx = obj.lift.econ_full( zeta );
                zxblock = kron( eye( obj.params.m ) , z );   % block matrix of z
                Beta = B * zxblock;
            else
%                 zx = obj.lift.econ_full_loaded( zeta , w );
                zxblock = kron( eye( obj.params.m ) , z );   % block matrix of z
                Beta = B * zxblock;
            end
        end
        
        % get_NLmodel (Builds discrete time nonlinear Koopman model)
        function [ out , obj ] = get_NLmodel( obj , koopData )
            %get_NLmodel: Defines the vector field F(x,u) that describe the nonlinear
            %discrete system x_k+1 = F(x_k,u_k), or x_k+1 = F(x_k,u_k,w_k)
            %   out - struct with fields 
            %       F_sym - symbolic expression of F
            %       F_func = function handle for function that evaluates F(x,u)
            %   obj.model - property of struct which stores the model
            
            % convert to continuous time Koopman matrix
            if strcmp( obj.time_type , 'continuous' )
                koopData.K_dis = koopData.K;
                koopData.K = (1/obj.params.Ts) * logm( koopData.K + 1e-12 * eye(size(koopData.K)) );   % apply conditioning
            end
            
            % define the unloaded basis vector
            if obj.dim_red  % reduced order basis
                basis_noload = [ obj.params.zeta ; obj.params.u ; obj.basis.econ ; 1 ];  % unlifted state and input
            else    % not reduced order basis
                basis_noload = obj.basis.full;
            end
            
            % define symbolic expression for dynamics
            if obj.loaded
                basis_loaded = basis_noload;
                for i = 1 : obj.params.nw
                    basis_loaded = [ basis_loaded ; obj.params.w(i) * basis_noload ]; % lifted state including loads
                end
                F = koopData.K( : , 1 : obj.params.nzeta )' * basis_loaded;
                out.F_sym = F;
                out.F_func = matlabFunction( F , 'Vars', { obj.params.zeta , obj.params.u , obj.params.w } );
            else
                F = koopData.K( : , 1 : obj.params.nzeta )' * basis_noload;
                out.F_sym = F;
                out.F_func = matlabFunction( F , 'Vars', {obj.params.zeta, obj.params.u} );
            end

            out.params = obj.params;    % save local copy of model parameters
            
            % define a C matrix as identity (needed for Kmpc class)
            out.C = eye( obj.params.n );
            
            % add model substruct to the class
            obj.model = out; 
        end
               
        % train_models (train multiple koopman models with diff. lasso params)
        function obj = train_models( obj , lasso )
            %train_model: finds koopman operator for given snapshot pairs
            %and lasso params.
            %   lasso - array containing lasso penalty values. The size of
            %     lasso determines whether the outputs will be cell arrays or
            %     not.
            
            % lasso argument is optional
            if nargin < 2
                lasso = obj.lasso;
            end
            
            if length(lasso) < 2
                obj.koopData = obj.get_Koopman( obj.snapshotPairs , lasso );
                if strcmp( obj.model_type , 'nonlinear' )
                    [ temp , obj ] = obj.get_NLmodel( obj.koopData );
                    obj.candidates = temp;
                elseif strcmp( obj.model_type , 'bilinear' )
                    [ temp , obj ] = obj.get_BLmodel( obj.koopData );
                    obj.candidates = temp; 
                else
                    obj.candidates = obj.get_model( obj.koopData );
                end
                obj.candidates.lasso = lasso;
                obj.model = obj.candidates; % save as the model
            else
                obj.koopData = cell( length(lasso) , 1 );
                obj.candidates = cell( length(lasso) , 1 );
                for i = 1 : length(lasso)
                    obj.koopData{i} = obj.get_Koopman( obj.snapshotPairs , lasso(i) );
                    if strcmp( obj.model_type , 'nonlinear' )
                        [ temp , obj ] = obj.get_NLmodel( obj.koopData{i} );
                        obj.candidates{i} = temp;
                    elseif strcmp( obj.model_type , 'bilinear' )
                        [ temp , obj ] = obj.get_BLmodel( obj.koopData{i} );
                        obj.candidates{i} = temp;
                    else
                        obj.candidates{i} = obj.get_model( obj.koopData{i} );
                    end
                    obj.candidates{i}.lasso = lasso(i);  % save lasso parameter with model
                    % save the first candidate as 'model' by default. Will be
                    %   overwritten by user choice later
                    obj.model = obj.candidates{1};
                end
            end
        end
        
        %% Reduce dimension of the set of basis functions (only works for poly and loaded systems)
        
        % lift_snapshots (lift the snapshots)
        function Px = lift_snapshots( obj , snapshotPairs )
            
            % Extract snapshot pairs
            [x,y,u] = deal( snapshotPairs.alpha , snapshotPairs.beta , snapshotPairs.u );
            if isfield( snapshotPairs , 'w' )
                w = snapshotPairs.w;
                nw = obj.params.nw;
            else
                nw = 0; % null nw for when there is no load
            end
            
            % Build matrices
            [~, m] = deal( obj.params.n , obj.params.m );
            Nm = obj.params.N + m;   % dimension of z plus input
            N = obj.params.N;       % dimension of z (i.e. lifted state)
            
            % preallocation
            if strcmp( obj.model_type , 'bilinear' )
                Px = zeros(length(x), N );
            else
                Px = zeros(length(x), N );
            end
            fprintf('Evaluating all basis functions on snapshots...\n');
            for i = 1:length(x)
                if ~mod( i , floor(length(x)/10) )  % only update progress once in a while
                    obj.loop_progress( i , length(x) );  % display progress
                end
                if strcmp( obj.model_type , 'nonlinear' )    % nonlinear lift includes the input 
                    psix = obj.lift.full( [ x(i,:) , u(i,:) ]' )';  
                    Px(i,:) = psix;
                elseif strcmp( obj.model_type , 'bilinear' )    % don't include input in this lift 
                    psix = obj.lift.full( x(i,:)' )';  
                    Px(i,:) = psix;
                else
                    psix = obj.lift.full( x(i,:)' )';   
                    Px(i,:) = psix;
                end
            end 
        end
        
        % get_econ_observables (get a lower dimensional set of observables)
        function obj = get_econ_observables( obj , Px )
            
            if ~obj.dim_red     % don't perform dimensional reduction
                % set "econ" versions of the lifting functions as copies
                if strcmp( obj.model_type , 'bilinear' )    % differnent because inputs must be kept out of dimensional reduction
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.lift.full(zeta_in);
                        obj.lift.econ_full_loaded = @(zeta_in,w_in)obj.lift.full_loaded(zeta_in,w_in);
                        obj.lift.econ_full_loaded_input = @(zeta_in,w_in,u_in)obj.lift.full_loaded_input(zeta_in,w_in,u_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                        obj.basis.econ_full_loaded = obj.basis.full_loaded;
                        obj.basis.econ_full_loaded_input = obj.basis.full_loaded_input;
                    else
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.lift.full(zeta_in);
                        obj.lift.econ_full_input =  @(zeta_in,u_in)obj.lift.full_input(zeta_in,u_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                        obj.basis.econ_full_input = obj.basis.full_input;
                    end
                elseif strcmp( obj.model_type , 'nonlinear' )    % differnent because inputs must be lifted with zeta
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(unlifted_in)obj.lift.full( unlifted_in);
                        obj.lift.econ_full_loaded = @(unlifted_in,w_in)obj.lift.full_loaded(unlifted_in,w_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                        obj.basis.econ_full_loaded = obj.basis.full_loaded;
                    else
                        % functions
                        obj.lift.econ_full = @(unlifted_in)obj.lift.full(unlifted_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                    end
                else    % case used for linear systems
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.lift.full(zeta_in);
                        obj.lift.econ_full_loaded = @(zeta_in,w_in)obj.lift.full_loaded(zeta_in,w_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                        obj.basis.econ_full_loaded = obj.basis.full_loaded;
                    else
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.lift.full(zeta_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.basis.full;
                    end
                end
                % Save a symbolic version of the reduced order basis
                obj.basis.econ = obj.basis.full(1:end-1);   % don't include constant at end
            else    % perform dimensional reduction
                % take pca of lifted snapshots
                [ coeffs , ~ , ~ , ~ , explained , ~ ] = pca( Px );
                
                % take enough components to explain > 99% of the data
                num_pcs = 1;
                while sum( explained(1:num_pcs) ) < 99
                    num_pcs = num_pcs + 1;
                end
                
                % extract first num_pcs columns
                pcs = coeffs( : , 1 : num_pcs );
                
                % Chenge the dimension parameters for the lifted state
                obj.basis.pcs = pcs;    % principal components
                if strcmp( obj.model_type , 'nonlinear' )
                    obj.params.N = num_pcs + obj.params.nzeta + obj.params.m + 1; % the dimension of the lifted state
                elseif strcmp( obj.model_type , 'bilinear' )
                    obj.params.N = num_pcs + obj.params.nzeta + 1; % the dimension of the lifted state, including loads
                else
                    obj.params.N = num_pcs + obj.params.nzeta + 1; % the dimension of the lifted state
                end
                
                % Save new reduced dimension lifting functions to the class
                if strcmp( obj.model_type , 'bilinear' )    % differnent because inputs must be kept out of dimensional reduction
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.econ_full(zeta_in);
                        obj.lift.econ_full_loaded = @(zeta_in,w_in)obj.econ_full_loaded(zeta_in,w_in);
                        obj.lift.econ_full_loaded_input = @(zeta_in,w_in,u_in)obj.econ_full_loaded_input(zeta_in,w_in,u_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( obj.params.zeta );
                        obj.basis.econ_full_loaded = obj.econ_full_loaded( obj.params.zeta , obj.params.w );
                        obj.basis.econ_full_loaded_input = obj.econ_full_loaded_input( obj.params.zeta , obj.params.w , obj.params.u );
                    else
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.econ_full(zeta_in);
                        obj.lift.econ_full_input =  @(zeta_in,u_in)obj.econ_full_input(zeta_in,u_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( obj.params.zeta );
                        obj.basis.econ_full_input = obj.econ_full_input( obj.params.zeta , obj.params.u );
                    end
                elseif strcmp( obj.model_type , 'nonlinear' )    % differnent because inputs must be lifted with zeta
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(unlifted_in)obj.econ_full( unlifted_in);
                        obj.lift.econ_full_loaded = @(unlifted_in,w_in)obj.econ_full_loaded(unlifted_in,w_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( [ obj.params.zeta ; obj.params.u ] );
                        obj.basis.econ_full_loaded = obj.econ_full_loaded( [ obj.params.zeta ; obj.params.u ] , obj.params.w );
                    else
                        % functions
                        obj.lift.econ_full = @(unlifted_in)obj.econ_full(unlifted_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( [ obj.params.zeta ; obj.params.u ] );
                    end
                else    % case used for linear systems
                    if obj.loaded
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.econ_full(zeta_in);
                        obj.lift.econ_full_loaded = @(zeta_in,w_in)obj.econ_full_loaded(zeta_in,w_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( obj.params.zeta );
                        obj.basis.econ_full_loaded = obj.econ_full_loaded( obj.params.zeta , obj.params.w );
                    else
                        % functions
                        obj.lift.econ_full = @(zeta_in)obj.econ_full(zeta_in);
                        
                        % symbolic expressions
                        obj.basis.econ_full = obj.econ_full( obj.params.zeta );
                    end
                end
                
                % Save a symbolic version of the reduced order basis
                obj.basis.econ = obj.basis.pcs' * obj.basis.full;
            end
        end
        
        % econ_full_loaded_input (for bilinear systems only)
        function zwecon = econ_full_loaded_input( obj , zeta , w , u )
            %econ_lift_bilinear: define new lifting function for lower dim basis
            %   zeta - unlifted state with input appended
            %   w - load
            
            gecon = obj.basis.pcs' * obj.lift.full( zeta ); % lift to full size
            gecon_diag = kron( eye(obj.params.nw+1) , [ zeta ; gecon ; 1 ] );   % constant appended at end
            full_loaded = gecon_diag * [ 1 ; w ];
            phi_diag = kron( eye(obj.params.m+1) , full_loaded );
            full_loaded_input = phi_diag * [ 1 ; u ];
            zwecon = full_loaded_input;
        end
        
        % econ_full_input (for bilinear systems only)
        function zecon = econ_full_input( obj , zeta , u )
            %econ_noload_bilinear: lift without the input
            %   zeta - unlifted state with input appended
            
            gecon = obj.basis.pcs' * obj.lift.full( zeta ); % lift to full size
%             g_diag = kron( eye(obj.params.nw+1) , [ zeta ; g ] );
%             full_loaded = g_diag * [ 1 ; w ];
            phi_diag = kron( eye(obj.params.m+1) , [ zeta ; gecon ; 1 ] );  % constant appended at end
            full_input = phi_diag * [ 1 ; u ];
            zecon = full_input;
        end
        
        % econ_full_loaded
        function zwecon = econ_full_loaded( obj , zeta , w )           
            gecon = obj.basis.pcs' * obj.lift.full( zeta );
            gecon_diag = kron( eye(obj.params.nw+1) , [ zeta ; gecon ; 1 ] );   % constant term appended
            full_loaded = gecon_diag * [ 1 ; w ];
            zwecon = full_loaded;
        end
        
        % econ_full
        function zecon = econ_full( obj , zeta )         
            gecon = obj.basis.pcs' * obj.lift.full( zeta );
            zecon = [ zeta ; gecon ; 1 ];   % constant term appended
        end
        
        %% validate performance of a fitted model
        
        % val_model (compares model simulation to real data)
        function results = val_model( obj , model , valdata )
            %val_model: Compares a model simulation to real data
            %   model - struct with fields A, B, C, sys, ...
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            
            % shift real data so delays can be accounted for
            index0 = obj.params.nd + 1;  % index of the first state
            treal = valdata.t(index0 : end);    % start simulation late so delays can be taken into account
            yreal = valdata.y(index0 : end , :);
            ureal = valdata.u(index0 : end , :);
            [ ~ , zetareal ] = obj.get_zeta( valdata );
            if obj.loaded 
                wreal = valdata.w(index0 : end , :);
                zreal = zeros( size( zetareal , 2 ) , obj.params.N * (obj.params.nw+1) );
                for i = 1 : size( zetareal , 1 )
                    zreal(i,:) = obj.lift.econ_full_loaded( zetareal(i,:)' , wreal(i,:)' );
                end
            else
                zreal = zeros( size( zetareal , 2 ) , obj.params.N );
                for i = 1 : size( zetareal , 1 )
                    zreal(i,:) = obj.lift.econ_full( zetareal(i,:)' );
                end
            end
            
            % simulate lifted bilinear linear model
            tsim = treal;
            usim = ureal;
            ysim = zeros( size( yreal ) ); % preallocate
            zsim = zeros( size( zreal ) ); % preallocate
            zetasim = zeros( size( zetareal ) );    % preallocate
            ysim(1,:) = yreal(1,:); % initialize
            zetasim(1,:) = zetareal(1,:)';    % initialize
            zeta0 = zetareal(1,:)';    % initial state with delays
            if obj.loaded
                wsim = wreal;   % load condition
                
                % set initial condition
                z0 = obj.lift.econ_full_loaded( zeta0 , wsim(1,:)' );    % initial lifted state
                zsim(1,:) = z0';        % initialize
                
                % simulate system response
                for j = 1 : length(treal)-1
                    % compute current lifted state using actual load condition
                    znow = kron( eye(obj.params.nw+1) , zsim(j,1:obj.params.N)' ) * [ 1 ; wsim(j,:)' ];
                    zsim(j+1,:) = ( model.A * znow + model.B * usim(j,:)' )';
                    zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                    ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                end
            else
                % set initial condition
                z0 = obj.lift.econ_full( zeta0 );    % initial lifted state
                zsim(1,:) = z0';        % initialize
                
                % simulate system response
                for j = 1 : length(treal)-1
                    if strcmp( obj.time_type , 'continuous' )
                        [ ~ , zt ] = ode45( @(t,z) model.A * z + model.B * usim(j,:)' , [0 , obj.params.Ts] , zsim(j,:)' );
                        zsim(j+1,:) = zt(end,:);
                        zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                        ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                    else
                        zsim(j+1,:) = ( model.A * zsim(j,:)' + model.B * usim(j,:)' )';
                        zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                        ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                    end
                end
            end
            
            % save simulations in output struct
            results = struct;
            results.t = treal; 
            
            results.sim.t = tsim;
            results.sim.u = usim;
            results.sim.y = ysim;
            results.sim.z = zsim;
            results.sim.zeta = zetasim;
            
            results.real.t = treal;
            results.real.u = ureal;
            results.real.y = yreal;
            results.real.z = zreal;
            results.real.zeta = zetareal;
            if obj.loaded
                results.sim.w = wsim;
                results.real.w = wreal;
            end
            
            % save error info (optional, could get rid of this)
            results.error = obj.get_error( results.sim , results.real );
        end
        
        % val_BLmodel (compares model simulation to real data)
        function results = val_BLmodel( obj , model , valdata )
            %val_model: Compares a model simulation to real data
            %   model - struct with fields A, B, C, Beta, ...
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            
            % shift real data so delays can be accounted for
            index0 = obj.params.nd + 1;  % index of the first state
            treal = valdata.t(index0 : end);    % start simulation late so delays can be taken into account
            yreal = valdata.y(index0 : end , :);
            ureal = valdata.u(index0 : end , :);
            [ ~ , zetareal ] = obj.get_zeta( valdata );
            if obj.loaded 
                wreal = valdata.w(index0 : end , :);
                zreal = zeros( size( zetareal , 2 ) , obj.params.N * (obj.params.nw+1) );
                for i = 1 : size( zetareal , 1 )
                    zreal(i,:) = obj.lift.econ_full_loaded( zetareal(i,:)' , wreal(i,:)' );
                end
            else
                zreal = zeros( size( zetareal , 2 ) , obj.params.N );
                for i = 1 : size( zetareal , 1 )
                    zreal(i,:) = obj.lift.econ_full( zetareal(i,:)' );
                end
            end
            
            % simulate lifted bilinear linear model
            tsim = treal;
            usim = ureal;
            ysim = zeros( size( yreal ) ); % preallocate
            zsim = zeros( size( zreal ) ); % preallocate
            zetasim = zeros( size( zetareal ) );    % preallocate
            ysim(1,:) = yreal(1,:); % initialize
            zetasim(1,:) = zetareal(1,:)';    % initialize
            zeta0 = zetareal(1,:)';    % initial state with delays
            if obj.loaded
                wsim = wreal;   % load condition
                
                % set initial condition
                z0 = obj.lift.econ_full_loaded( zeta0 , wsim(1,:)' );    % initial lifted state
                zsim(1,:) = z0';        % initialize
                
                % simulate system response
                for j = 1 : length(treal)-1
                    % compute current lifted state using actual load condition
                    znow = kron( eye(obj.params.nw+1) , zsim(j,1:obj.params.N)' ) * [ 1 ; wsim(j,:)' ];
                    zsim(j+1,:) = ( model.A * znow + model.Beta( znow ) * usim(j,:)' )';
                    zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                    ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                end
            else
                % set initial condition
                z0 = obj.lift.econ_full( zeta0 );    % initial lifted state
                zsim(1,:) = z0';        % initialize
                
                % simulate system response
                for j = 1 : length(treal)-1
%                     % debug
%                     zsim_winputs = obj.lift.econ_full_input( zetasim(j,:)' , usim(j,:)' );
%                     zsim_winputs_plus = obj.model.K' * zsim_winputs;
%                     zsim(j+1,:) = zsim_winputs_plus( 1 : obj.params.N )';
                    if strcmp( obj.time_type , 'continuous' )
                        [ ~ , zt ] = ode45( @(t,z) model.A * z + model.Beta(z) * usim(j,:)' , [0 , obj.params.Ts] , zsim(j,:)' );
                        zsim(j+1,:) = zt(end,:);
                        zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                        ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                    else
                        zsim(j+1,:) = ( model.A * zsim(j,:)' + model.Beta( zsim(j,:)' ) * usim(j,:)' )';
                        zetasim(j+1,:) = zsim( j+1 , 1 : obj.params.nzeta );
                        ysim(j+1,:) = ( model.C * zsim(j+1,:)' )';
                    end
                end
            end
            
            % save simulations in output struct
            results = struct;
            results.t = treal; 
            
            results.sim.t = tsim;
            results.sim.u = usim;
            results.sim.y = ysim;
            results.sim.z = zsim;
            results.sim.zeta = zetasim;
            
            results.real.t = treal;
            results.real.u = ureal;
            results.real.y = yreal;
            results.real.z = zreal;
            results.real.zeta = zetareal;
            if obj.loaded
                results.sim.w = wsim;
                results.real.w = wreal;
            end
            
            % save error info (optional, could get rid of this)
            results.error = obj.get_error( results.sim , results.real );
        end
        
        % val_NLmodel (compares model simulation to real data)
        function results = val_NLmodel( obj , model , valdata )
            %val_NLmodel: Compares a model simulation to real data
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            
            % shift real data so delays can be accounted for
            index0 = obj.params.nd + 1;  % index of the first state
            treal = valdata.t(index0 : end);    % start simulation late so delays can be taken into account
            yreal = valdata.y(index0 : end , :);
            ureal = valdata.u(index0 : end , :);
            [ ~ , zetareal ] = obj.get_zeta( valdata );
            zreal = zeros( size( zetareal , 2 ) , obj.params.N );
            for i = 1 : size( zetareal , 1 )
                zreal(i,:) = obj.lift.econ_full( [ zetareal(i,:) , ureal(i,:) ]' );
            end
            
            % include load if it exists
            results = struct;
            if obj.loaded
                wreal = valdata.w(index0 : end , :);
                wsim = wreal;
                results.real.w = wreal;
                results.sim.w = wsim;
            end
            
            % set initial condition
            zetasim = zeros( size( zetareal ) ); % preallocate
            zetasim( 1 , : ) = zetareal( 1, : );
            
            % simulate nonlinear model
            tsim = treal;
            usim = ureal;
%             [ tsim , zetasim ] = ode45( @(t,y) model.F_func( y , ureal( floor(t/obj.params.Ts)+1 ,:)' ) , treal , zeta0 );
            for j = 1 : length(treal)-1
                if strcmp( obj.time_type , 'continuous' )   % continuous time model
                    if obj.loaded
                        [ ~ , zetat ] = ode45( @(t,zeta) model.F_func( zeta , usim(j,:)' , wsim(j,:)' ) , [0 , obj.params.Ts] , zetasim(j,:)' );
                    else
                        [ ~ , zetat ] = ode45( @(t,zeta) model.F_func( zeta , usim(j,:)' ) , [0 , obj.params.Ts] , zetasim(j,:)' );
                    end
                    zetasim(j+1,:) = zetat(end,:);
                else    % discrete time model
                    if obj.loaded
                        zetasim(j+1,:) = model.F_func( zetasim(j,:)' , usim(j,:)' , wsim(j,:)' );
                    else
                        zetasim(j+1,:) = model.F_func( zetasim(j,:)' , usim(j,:)' );
                    end
                end
            end
            ysim = zetasim( : , 1 : obj.params.n );
            
            % save simulations in output struct
            results.t = treal;
            
            results.sim.t = tsim;
            results.sim.u = usim;
            results.sim.y = ysim;
            
            results.real.t = treal;
            results.real.u = ureal;
            results.real.y = yreal;
            
            % save error info (optional, could get rid of this)
            results.error = obj.get_error( results.sim , results.real );
        end
        
        % get_error (computes the error between real and simulated data)
        function err = get_error( obj , simdata , realdata )
            %get_error: Computes the error between real and simulated data.
            
            % scaled down data in [-1,1]
            err.abs = abs( simdata.y - realdata.y );  % absolute error over time
            err.mean = mean( err.abs , 1 );   % average absolute error over time
            err.rmse = sqrt( sum( (simdata.y - realdata.y).^2 , 1 ) ./ length(realdata.t) ); % RMSE (over each state)
            err.nrmse = err.rmse ./ abs( max( realdata.y ) - min( realdata.y ) );   % RMSE normalized by total range of real data values
            err.euclid = sqrt( sum( (simdata.y - realdata.y).^2 , 2 ) );    % euclidean distance error in R^n at each point
            err.euclid_mean = sum( err.euclid ) / length(realdata.t);   % average euclidean distance error over all time steps
            
            % data in original data range
            ysim = obj.scaleup.y( simdata.y );
            yreal = obj.scaleup.y( realdata.y );
            err.unscaled.euclid = sqrt( sum( (ysim - yreal).^2 , 2 ) );    % euclidean distance error in R^n at each point
            err.unscaled.euclid_mean = sum( err.unscaled.euclid ) / length(realdata.t);   % average euclidean distance error over all time steps
        end
        
        % plot_comparison (plots a comparison between simulation and real data)
        function plot_comparison( obj , simdata , realdata , figtitle)
            %plot_comparison: plots a comparison between simulation and real data.
            
            % quantify the error
            err = obj.get_error( simdata , realdata );
            
            % create new figure
            if nargin > 3
                figure('NumberTitle', 'off', 'Name', figtitle);  
            else
                figure;
            end
            
            for i = 1 : obj.params.n
                subplot( obj.params.n , 1 , i );
                ylabel( [ 'y' , num2str(i) ] );
                title( [ 'NRMSE = ' , num2str( err.nrmse(i) ) ] );
                ylim([-1,1]);
                hold on;
                plot( realdata.t , realdata.y( : , i ) , 'b' );
                plot( simdata.t , simdata.y( : , i ) , 'r' );
                hold off;
            end
            legend({'Real' , 'Koopman'});
        end
        
        % valNplot_model (run val_model and plot_comparison for a given model)
        function [ results , err ] = valNplot_model( obj , model_id , saveask )
            %valNplot_model: run val_model and plot_comparison for a given model
            
            if nargin < 3
                saveask = true;    % ask if user wants to save the model
            end
            
            % if no model id is provided, use first one in the model cell array
            if nargin < 2 && ~iscell(obj.candidates)
                mod = obj.candidates;
            elseif nargin < 2
                mod = obj.candidates{1};
            elseif isempty( model_id )
                mod = obj.candidates;
            else
                mod = obj.candidates{model_id};
            end
                
            results = cell( size(obj.valdata) );    % store results in a cell array
            err = cell( size(obj.valdata) );    % store error in a cell array
            for i = 1 : length(obj.valdata)
                if strcmp( obj.model_type , 'nonlinear' )
                    results{i} = obj.val_NLmodel( mod , obj.valdata{i} );
                elseif strcmp( obj.model_type , 'bilinear' )
                    results{i} = obj.val_BLmodel( mod , obj.valdata{i} );
                else
                    results{i} = obj.val_model( mod , obj.valdata{i} );
                end
                err{i} = obj.get_error( results{i}.sim , results{i}.real );
                obj.plot_comparison( results{i}.sim , results{i}.real , ['Lasso: ' , num2str(mod.lasso)] );
            end
            
            % save (or don't save) sysid class, model, and training data
            if saveask
                saveModel = questdlg( 'Would you like to save this model?' , '' , 'Yes' , 'Not right now' , 'Not right now' );
                if strcmp( saveModel , 'Yes' )
                    obj.save_class;
                end
            end
        end
        
        
        %% infer the load based on learned dynamics
        
        % observer_load (infer the load based on dynamics)
        function [ what , resnorm ] = observer_load( obj , ypast , upast , whatpast )
            % observer_load: Estimate the load given measurements over a 
            % past horizon.
            %   ypast - [hor x n], output measurements over previous hor steps
            %   upast - [hor x m], inputs over previous hor steps
            %   whatpast - [1 x nw], load estimate at previous step (optional)
            %   resnorm - squared 2-norm of residual of cost function
            % Note: This doesn't work for delays yet...
            
            hor = size( ypast , 1 ); % length of past horizon
            if size(upast,1) ~= hor
                error('Input arguments must have the same number of rows');
            end
            
            % stack Omega and u vertically
            Omega = zeros( obj.params.N * (obj.params.nw+1) * (hor-1) , obj.params.nw+1 );
            for i = 1 : hor-1
                Omega_i = obj.lift.Omega( ypast(i,:)' );
                ind1 = (obj.params.N*(obj.params.nw+1))*(i-1)+1;
                ind2 = (obj.params.N*(obj.params.nw+1))*i;
                Omega( ind1 : ind2 , : ) = Omega_i; 
            end
            U = reshape( upast(1:end-1,:)' , [ obj.params.m * (hor-1) , 1 ] );
            Y = reshape( ypast( 2:end , 1:obj.params.n )' , [ obj.params.n * (hor-1) , 1 ] );
%             Y = reshape( ypast(2:end,:)' , [ obj.params.nzeta * (hor-1) , 1 ] );    % make dimensions of y consistent with delays
            
            % cost function matrices
            CAstack = kron(eye(hor-1) , obj.model.C * obj.model.A);
            CBstack = kron(eye(hor-1) , obj.model.C * obj.model.B);
            Clsqlin = CAstack * Omega;
            dlsqlin = Y - CBstack * U;
            
            % optional: make sure new load estimate is close to last one
            if nargin < 4
                A = zeros( obj.params.nw + 1 , obj.params.nw + 1 );
                b = zeros( obj.params.nw + 1 , 1 );
            else
                % inequality contsraints (acts as slope constraint)
                A = [ -whatpast(end,:)' , eye( obj.params.nw );...
                    whatpast(end,:)' , -eye( obj.params.nw )];
                b = 0.01 * ones( obj.params.nw + 1 , 1 );
            end
            
            % equality constraint matrices
            Aeq = blkdiag( 1 , zeros(obj.params.nw , obj.params.nw) );
            beq = [ 1 ; zeros(obj.params.nw,1) ]; % ensure first elements is 1
            lb = -ones(obj.params.nw+1,1);  % load should be in [-1,1]
            ub = ones(obj.params.nw+1,1);   % load should be in [-1,1]
            
            % solve for what
            [ sol , resnorm ] = lsqlin( Clsqlin , dlsqlin , A , b , Aeq , beq , lb , ub );  % solve for what using constrained least squares solver
            what = sol(2:end);
        end
        
        % val_observer_load (evaluate the accuracy of observer)
        function [ what , wreal , werr ] = val_observer_load( obj , hor , valdata )
            %val_observer_load: Compares load estimates to real data
            %   hor - length of backward looking horizon
            %   model - struct with fields A, B, C, sys, ...
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            %
            %   Notation is weird, yhor should actually be called zetahor
            %   so that delays can be incorporated, but I didn't feel like
            %   changing everything.
            
            what = zeros( length( valdata.t ) , obj.params.nw );
%             yhor = zeros( hor , obj.params.n );
            yhor = zeros( hor , obj.params.nzeta );     % make y work with delays
            uhor = zeros( hor , obj.params.m );
            [ ~ , zeta ] = obj.get_zeta( valdata );
            for i = 1 : length( valdata.t ) - 1
                % Measure current output (with some noise)
                mu = 0; % mean of measurement noise
                sigma = 0.00;  % 0.005 standard deviation of measurement noise
%                 y_i = valdata.y(i,:) + normrnd(mu,sigma);
                y_i = zeta(i,:) + normrnd(mu,sigma);
                u_i = valdata.u(i,:);
                
                % Integrate new measurement into past horizon of measurements
                yhor = [ yhor(2:end,:) ; y_i ];   % newest at end
                uhor = [ uhor(2:end,:) ; u_i ];   % newest at end
                
%                 ysmooth = smoothdata(yhor); % this may be slow, consider replacing
                ysmooth = yhor;
                
                % Estimate the load
%                 what(i+1,:) = obj.observer_load( ysmooth , uhor , what(i,:) )'; % with max change limitation
                what(i+1,:) = obj.observer_load( ysmooth , uhor )';
%                 what_new = obj.observer_load( ysmooth , uhor )';
                
%                 % record the average of the new estimate and the previous one
%                 what(i+1,:) = 0.5 * ( what(i,:) + what_new );
            end
            
            % Compare estimate to real load
            wreal = valdata.w;
            werr = abs( wreal - what );
        end
        
        % val_observer_load_sparse (evaluate the accuracy of observer)
        function [ what , wreal , werr , res ] = val_observer_load_sparse( obj , hor , update_hor , valdata )
            %val_observer_load_sparse: Compares load estimates to real data
            %   hor - length of backward looking horizon
            %   model - struct with fields A, B, C, sys, ...
            %   valdata - struct with fields t, y, u (at least)
            %   results - struct with simulation results and error calculations
            %   updata_hor - length of time to wait before a new update
            %
            %   Notation is weird, yhor should actually be called zetahor
            %   so that delays can be incorporated, but I didn't feel like
            %   changing everything.
            %
            %   Same as val_observer_load, except it only updates the
            %   estimate every 'update_hor' 
            
            what_all = [];  % not memory efficient, but just go with for now
            what = zeros( length( valdata.t ) , obj.params.nw );
            res = zeros( length( valdata.t ) , 1 ) + 1e-6;
%             yhor = zeros( hor , obj.params.n );
            yhor = zeros( hor , obj.params.nzeta );     % make y work with delays
            uhor = zeros( hor , obj.params.m );
            [ ~ , zeta ] = obj.get_zeta( valdata );
            for i = 1 : length( valdata.t ) - 1
                % Measure current output (with some noise)
                mu = 0; % mean of measurement noise
                sigma = 0.00;  % 0.005 standard deviation of measurement noise
                y_i = zeta(i,:) + normrnd(mu,sigma);
                u_i = valdata.u(i+1,:);
                
                % Integrate new measurement into past horizon of measurements
                yhor = [ yhor(2:end,:) ; y_i ];   % newest at end
                uhor = [ uhor(2:end,:) ; u_i ];   % newest at end
                
%                 ysmooth = smoothdata(yhor); % this may be slow, consider replacing
                ysmooth = yhor;
                
                if mod( i , update_hor ) == 0
                    % Estimate the load
                    [ what_new_transpose , res_new ] = obj.observer_load( ysmooth , uhor );
                    what_new = what_new_transpose';
                    res_new = res_new + 1e-6;   % prevent zeros
                    
%                     % record the average of the new estimate and the previous one
%                     weight = res_new / ( res_new + res(i,:) );
%                     what(i+1,:) = (weight) * what(i,:) + (1 - weight) * what_new;
%                     res(i+1,:) = res_new;
                    
                    % record the average of the new estimate and all previous
                    what_all = [ what_all ; what_new ];
                    what(i+1,:) = mean( what_all , 1 );
                    res(i+1,:) = res_new;
                else
                    what(i+1,:) = what(i,:);
                    res(i+1,:) = res(i,:);
                end
            end
            
            % Compare estimate to real load
            wreal = valdata.w;
            werr = abs( wreal - what );
        end
        
    end
end
























