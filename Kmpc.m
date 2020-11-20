classdef Kmpc
    %kmpc: Model predictive controller class
    %   Detailed explanation goes here
    
    properties
        params; % paramaters of the system
        model;  % linear model of the system
        lift;   % lifting functions for system
        basis;  % symbolic basis set of observables
        model_type; % specified by the sysid_class
        loaded; % does model have loads incorporated into it?
        horizon;
        load_obs_horizon;   % backwards looking load observer horizon
        load_obs_period;    % timesteps between observer estimate updates
        input_bounds;
        input_slopeConst;
        input_smoothConst;
        state_bounds;
        cost_running;
        cost_terminal;
        cost_input;
        projmtx; % projection matrix from liftes state (z) to reference state   
        cost;   % stores cost matrices
        constraints;    % stores constraint matrices
        set_constRHS;  % function that sets the value of the RHS of constraints
        get_zeta;   % function that constructs zeta from state and input data in time
        
%         nlmpc_controller;   % matlab nlmpc controller object (MPC toolbox)
        
        scaledown;  % functions for scaling to [-1,1]
        scaleup;    % functions for scaling from [-1,1]
    end
    
    methods
        % CLASS CONSTRUCTOR
        function obj = Kmpc( sysid_class , varargin )
            %kmpc: Construct an instance of this class
            %   sysid_class - sysid class object with a model and params
            %    properties
            %   varargin - Name, Value pairs for class properties
            
            % take some properties/methods from the sysid class
            obj.params = sysid_class.params;
            obj.model = sysid_class.model;
            obj.lift = sysid_class.lift;
            obj.basis = sysid_class.basis;
            obj.get_zeta = @sysid_class.get_zeta;   % copies this method for convenience
            obj.scaledown = sysid_class.scaledown;
            obj.scaleup = sysid_class.scaleup;
            obj.model_type = sysid_class.model_type;    % 'linear' , 'bilinear' , 'nonlinear'
            obj.loaded = sysid_class.loaded;    % true or false
            
            % define default values of properties
            obj.horizon = floor( 1 / obj.params.Ts );
            obj.input_bounds = [];  % [min,max] can be 1x2 or mx2
            obj.input_slopeConst = [];
            obj.input_smoothConst = [];
            obj.state_bounds = []; % [min,max] can be 1x2 or nx2 
            obj.cost_running = 0.1;
            obj.cost_terminal = 100;
            obj.cost_input = 0;
            obj.projmtx = obj.model.C;   % recovers measured state (could also use Cshape)
            obj.cost = [];
            obj.constraints = [];
            obj.load_obs_horizon = 10;  % backwards looking load estimation horizon
            obj.load_obs_period = 1;    % update period for load estimation
            
            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );
            
            % define scaling funcions for reference trajectories based on projmtx
            obj.scaledown.ref = @(ref)obj.scaledown_ref(ref);   % assumes ref is a subset of y 
            obj.scaleup.ref = @(ref_sc)obj.scaleup_ref(ref_sc);     % assumes ref is a subset of y
            
            % resize some properties if they aren't full size already
            obj = obj.expand_props;
            
            % get cost and constraint matrices
            if strcmp( obj.model_type , 'linear' )
                obj = obj.get_costMatrices;
                obj = obj.get_constraintMatrices;
            elseif strcmp( obj.model_type , 'bilinear' )
                obj = obj.get_costMatrices_bilinear;
                obj = obj.get_constraintMatrices_bilinear;
            elseif strcmp( obj.model_type , 'nonlinear' )
                obj = obj.get_costMatrices_nonlinear;
                obj = obj.get_constraintMatrices_nonlinear;
            end
        end
        
        % parse_args: Parses the Name, Value pairs in varargin
        function obj = parse_args( obj , varargin )
            %parse_args: Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.(varargin{idx}) = varargin{idx+1} ;
            end
        end
        
        % expand_props: Converts props from shorthand to fully defined
        function obj = expand_props( obj )
            %expand_props: Converts props from shorthand to fully defined
            %   e.g. input_bounds = [ -Inf , Inf ] but params.m = 3,
            %   ==> [ -Inf , Inf ; -Inf , Inf ; -Inf , Inf ]
            
            % input_bounds
            if ~isempty( obj.input_bounds ) && size( obj.input_bounds , 1 ) ~= obj.params.m
                obj.input_bounds = kron( ones( obj.params.m , 1 ) , obj.input_bounds );
            end
            
            % state_bounds
            if ~isempty( obj.state_bounds ) && size( obj.state_bounds , 1 ) ~= obj.params.n
                obj.state_bounds = kron( ones( obj.params.n , 1 ) , obj.state_bounds );
            end     
        end
        
        %% Misc. functions
        
        % scaledown_ref: Scale down reference trajectories
        function ref_sc = scaledown_ref( obj , ref )
            proj_y2ref = obj.projmtx( : , 1 : obj.params.n ); % projection matrix from y to ref
            temp = zeros( size(ref,1) , obj.params.n ); % initialize
            ref_ind = sum( proj_y2ref , 1 );    % collapse into index vector
            temp( : , find(ref_ind) ) = ref;
            temp_sc = obj.scaledown.y( temp );
            ref_sc = temp_sc( : , find(ref_ind) );
        end
        
        % scaleup_ref: Scale up reference trajectories
        function ref = scaleup_ref( obj , ref_sc )
            proj_y2ref = obj.projmtx( : , 1 : obj.params.n ); % projection matrix from y to ref
            temp = zeros( size(ref_sc,1) , obj.params.n ); % initialize
            ref_ind = sum( proj_y2ref , 1 );    % collapse into index vector
            temp( : , find(ref_ind) ) = ref_sc;
            temp_sc = obj.scaleup.y( temp );
            ref = temp_sc( : , find(ref_ind) );
        end
        
        %% linear MPC functions
        
        % get_costMatrices: Contructs the matrices for the mpc optim. problem
        function obj = get_costMatrices( obj )
            %get_costMatrices: Constructs cost the matrices for the mpc 
            % optimization problem.
            %   obj.cost has fields H, G, D, A, B, C, Q, R
            
            % define cost function matrices
            % Cost function is defined: U'HU + ( z0'G + Yr'D )U
            
            model = obj.model;
            
            % A
            N = size(model.A,1);
            A = sparse( N*(obj.horizon+1) , N );
            for i = 0 : obj.horizon
                A( (N*i + 1) : N*(i+1) , : ) = model.A^i ;
            end
            
            % B
            Bheight = N*(obj.horizon+1);
            Bcolwidth = size(model.B,2);
            Bcol = sparse( Bheight , Bcolwidth );    % first column of B matrix
            for i = 1 : obj.horizon
                Bcol( (N*i + 1) : N*(i+1) , : ) = model.A^(i-1) * model.B ;
            end
            
            Lshift = spdiags( ones( N*obj.horizon , 1 ) , -N , N*(obj.horizon+1) , N*(obj.horizon+1) );    % lower shift operator
            
            Bwidth = size(model.B,2)*(obj.horizon);    % total columns in B matrix
            Bblkwidth = obj.horizon;   % how many Bcol blocks wide B matrix is
            B = spalloc( Bheight , Bwidth , floor(Bheight * Bwidth / 2) ); % initialze sparse B matrix
            B(: , 1:Bcolwidth) = Bcol;
            for i = 2 : Bblkwidth
                B(: , (i-1)*Bcolwidth+1 : i*Bcolwidth) = Lshift * B(: , (i-2)*Bcolwidth+1 : (i-1)*Bcolwidth);
            end
            
            % C: matrix that projects lifted state into reference trajectory space
            C = kron( speye(obj.horizon+1) , obj.projmtx);
            nproj = size( obj.projmtx , 1 );
            
            % Q: Error magnitude penalty
            Q = kron( speye(obj.horizon+1) , eye(nproj) * obj.cost_running); % error magnitude penalty (running cost) (default 0.1)
            Q(end-nproj+1 : end , end-nproj+1 : end) = eye(nproj) * obj.cost_terminal;    % (terminal cost) (default 100)
            
            % R: Input magnitude penalty
            R = kron( speye(obj.horizon) , eye(model.params.m) .* obj.cost_input );  % input magnitude penalty (for flaccy use 0.5e-2) (new videos used 0.5e-3)

            % H, G, D
            H = B' * C' * Q * C * B + R;
            G = 2 * A' * C' * Q * C * B;
            D = -2 * Q * C * B;
            
            % set outputs
            obj.cost.H = H; obj.cost.G = G; obj.cost.D = D; % constructed matrices
            obj.cost.A = A; obj.cost.B = B; obj.cost.C = C; obj.cost.Q = Q; obj.cost.R = R; % component matrices
        end
        
        % get_constraintMatrices: Constructs the constraint matrices
        function obj = get_constraintMatrices( obj )
            %get_constraintMatrices: Constructs the constraint matrices for
            % the mpc optimization problem.
            %   obj.constraints has fields L, M, F, E, (c?)
            %   F is for input constraints
            %   E is for state constraints
            
            % shorten some variable names
            Np = obj.horizon;     % steps in horizon
            params = obj.params;    % linear system model parameters
            cost = obj.cost;     % cost matrices
            
            F = []; E = [];     % initialize empty matrices
            c = [];
            
            % input_bounds
            if ~isempty( obj.input_bounds )
                num = 2*params.m;     % number of input bound constraints
                
                % F: input_bounds
                Fbounds_i = [ -speye(params.m) ; speye(params.m) ];    % diagonal element of F, for bounded inputs
                Fbounds = sparse( num * (Np+1) , size(cost.B,2) );  % no constraints, all zeros
                Fbounds( 1:num*Np , 1:Np*params.m ) = kron( speye(Np) , Fbounds_i );     % fill in nonzeros
                F = [ F ; Fbounds ];    % append matrix
                
                % E: input_bounds (just zeros)
                Ebounds = sparse( num * (Np+1) , size(cost.B,1) );  % no constraints, all zeros
                E = [ E ; Ebounds ];    % append matrix
                
                % c: input_bounds
                if isfield( obj.params , 'NLinput' )    % don't scale input if it's nonlinear
                    input_bounds_sc = obj.input_bounds;
                else
                    input_bounds_sc = obj.scaledown.u( obj.input_bounds' )';   % scaled down the input bounds
                end
                cbounds_i = [ -input_bounds_sc(:,1) ; input_bounds_sc(:,2) ]; % [ -umin ; umax ]
                cbounds = zeros( num * (Np+1) , 1);    % initialization
                cbounds(1 : num*Np) = kron( ones( Np , 1 ) , cbounds_i );     % fill in nonzeros
                c = [ c ; cbounds ];    % append vector
            end
            
            % input_slopeConst
            if ~isempty( obj.input_slopeConst )
                % F: input_slopeConst
                Fslope_i = speye(params.m);
                Fslope_neg = [ kron( speye(Np-1) , -Fslope_i ) , sparse( params.m * (Np-1) , params.m ) ];
                Fslope_pos = [ sparse( params.m * (Np-1) , params.m ) , kron( speye(Np-1) , Fslope_i ) ];
                Fslope_top = Fslope_neg + Fslope_pos;
                Fslope = [ Fslope_top ; -Fslope_top];
                F = [ F ; Fslope ];     % append matrix

                % E: input_slopeConst (just zeros)
                E = [ E ; sparse( 2 * params.m * (Np-1) , size(cost.B,1) ) ];
                
                % c: input_slopeConst
                if isfield( obj.params , 'NLinput' )    % don't scale slope if it's nonlinear
                    slope_lim = obj.input_slopeConst;
                else
                    slope_lim = obj.input_slopeConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                end
                cslope_top = slope_lim * ones( params.m * (Np-1) , 1 );
                cslope = [ cslope_top ; cslope_top ];
                c = [ c ; cslope ];     % append vector
            end
            
            % input_smoothConst
            if ~isempty( obj.input_smoothConst )
                % F: input_smoothConst
                Fsmooth_i = speye(params.m);
                Fsmooth_lI = [ kron( speye(Np-2) , Fsmooth_i ) , sparse( params.m * (Np-2) , 2 * params.m ) ];
                Fsmooth_2I = [ sparse( params.m * (Np-2) , params.m ) , kron( speye(Np-2) , -2*Fslope_i ) , sparse( params.m * (Np-2) , params.m ) ];
                Fsmooth_rI = [ sparse( params.m * (Np-2) , 2 * params.m ) , kron( speye(Np-2) , Fslope_i ) ];
                Fsmooth_top = Fsmooth_lI + Fsmooth_2I + Fsmooth_rI;
                Fsmooth = [ Fsmooth_top ; -Fsmooth_top ];
                F = [ F ; Fsmooth ];
                
                % E: input_smoothConst
                E = [ E ; sparse( 2 * params.m * (Np-2) , size(cost.B,1) ) ];
                
                % c: input_smoothConst
                smooth_lim = params.Ts^2 * obj.input_smoothConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                csmooth = smooth_lim * ones( size(Fsmooth,1) ,1);
                c = [ c ; csmooth ];
            end
            
            % state_bounds
            if ~isempty( obj.state_bounds )
                num = 2*params.n;   % number of state bound constraints
                
                % E: state_bounds
                Esbounds_i = [ -speye(params.n) ; speye(params.n) ];    % diagonal element of E, for bounding low dim. states (first n elements of lifted state)
                Esbounds = sparse( num * (Np+1) , size(cost.A,1) );  % no constraints, all zeros
                Esbounds( 1:num*(Np+1) , 1:(Np+1)*params.n ) = kron( speye(Np+1) , Esbounds_i );     % fill in nonzeros
                E = [ E ; Esbounds ];    % append matrix
                
                % F: state_bounds (all zeros)
                Fsbounds = zeros( size( Esbounds , 1 ) , size( cost.B , 2 ) );
                F = [ F ; Fsbounds ];    % append matrix
                
                % c: state_bounds
                state_bounds_sc = obj.scaledown.y( obj.state_bounds' )';    % scaled down state bounds
                csbounds_i = [ -state_bounds_sc(:,1) ; state_bounds_sc(:,2) ]; % [ -ymin ; ymax ]
                csbounds = kron( ones( Np+1 , 1 ) , csbounds_i );     % fill in nonzeros
                c = [ c ; csbounds ];    % append vector
            end
            
            % set outputs
            obj.constraints.F = F;
            obj.constraints.E = E;    
            obj.constraints.c = c;
            obj.constraints.L = F + E * cost.B;
            obj.constraints.M = E * cost.A;
        end
        
        % get_mpcInput: Solve the mpc problem to get the input over entire horizon
        function [ U , z ]= get_mpcInput( obj , traj , ref )
            %get_mpcInput: Soves the mpc problem to get the input over
            % entire horizon.
            %   traj - struct with fields y , u , (what). Contains the measured
            %    states, inputs, (and loads) for the past ndelay+1 timesteps.
            %   ref - matrix containing the reference trajectory for the
            %    system over the horizon (one row per timestep).
            %   z - the lifted state at the current timestep
            
            % shorthand variable names
            Np = obj.horizon;       % steps in the horizon
            nd = obj.params.nd;     % number of delays
            
            % construct the current value of zeta
            [ ~ , zeta_temp ] = obj.get_zeta( traj );
            zeta = zeta_temp( end , : )';   % want most recent points
            
            % lift zeta
            if obj.loaded
                z = obj.lift.econ_full_loaded( zeta , traj.what(end,:)' );
            else
                z = obj.lift.econ_full( zeta );
            end
            
            % check that reference trajectory has correct dimensions
            if size( ref , 2 ) ~= size( obj.projmtx , 1 )
                error('Reference trajectory is not the correct dimension');
            elseif size( ref , 1 ) > Np + 1
                ref = ref( 1 : Np + 1 , : );    % remove points over horizon
            elseif size( ref , 1 ) < Np + 1
                ref_temp = kron( ones( Np+1 , 1 ) , ref(end,:) );
                ref_temp( 1 : size(ref,1) , : ) = ref;
                ref = ref_temp;     % repeat last point for remainer of horizon
            end
            
            % vectorize the reference trajectory
            Yr = reshape( ref' , [ ( Np + 1 ) * size(ref,2) , 1 ] );
            
            % setup matrices for gurobi solver
            H = obj.cost.H;     
            f = ( z' * obj.cost.G + Yr' * obj.cost.D )';
            A = obj.constraints.L;
            b = - obj.constraints.M * z + obj.constraints.c;
            
            % tack on "memory" constraint to fix initial input u_0
            Atack = [ [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - obj.params.m ) ];
%             Atack_bot = [ sparse( 2*obj.params.m , obj.params.m) , [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - 2*obj.params.m ) ];
%             Atack = [ Atack_top ; Atack_bot ];
            btack = [ traj.u(end,:)' ; -traj.u(end,:)' ];
            A = [A ; Atack];    % tack on memory constraint
            b = [b ; btack];
            
            % solve the MPC problem
%             Uvec = quadprog_gurobi( H , f , A , b );   % solve using gurobi (returns NaNs of cannot be solved)
            Uvec = quadprog( 2*H , f , A , b );     % solve using matlab
            
            % reshape the output so each input will have one row (first row equals current input)
            U = reshape( Uvec , [ obj.params.m , Np ] )';
        end
        
        % resample_ref: Resamples a reference trajectory
        function ref_resampled = resample_ref( obj, ref )
            %resample_ref: Resamples a reference trajectory at the system
            % sampling time.
            % ref - struct with fields:
            %   t - time vector
            %   y - trajectory vector
            
            tr = 0 : obj.params.Ts : ref.t(end);
            ref_resampled = interp1( ref.t , ref.y , tr );
        end
        
        % run_simulation: Runs a simulation of system under mpc controller
        % (CONSIDER REMOVING THIS FUNCTION)
        function results = run_simulation( obj , ref , y0 , u0)
            %run_trial: Runs a simulation of system under mpc controller.
            %   Tries to follow the trajectory in ref and impose the
            %   shape constraints in shape_bounds.
            %   Assume ref and shape_bounds have same sampling frequency as
            %   sytem, and that they are already scaled to be consistent 
            %   with the lifted model.
            %   ref - struct containing reference trajectory with fields:
            %       t - vector of timesteps
            %       y - each row is a desired point at the corresponding timestep
            %   x0 - [1,n] initial condtion
            %   u0 - [1,m] initial input
            
            % shorthand
            nd = obj.params.nd;
            Np = obj.horizon;
            
            % set value of initial conditions to zero if none provided
            if nargin < 3
                y0 = zeros( nd+1 , obj.params.n );
                u0 = zeros( nd+1 , obj.params.m );
            elseif nargin < 4
                y0 = kron( ones( nd+1 , 1 ) , y0 );
                u0 = zeros( nd+1 , obj.params.m );
            else
                y0 = kron( ones( nd+1 , 1 ) , y0 );
                u0 = kron( ones( nd+1 , 1 ) , u0 );
            end
            
            % resample and scale the reference trajectory
            ref_Ts = obj.resample_ref( ref );
            ref_sc = obj.scaledown.y( ref_Ts );
            
            % set initial condition
            initial.y = y0; initial.u = u0;
            [ initial , zeta0 ] = obj.get_zeta( initial );    % LINE NOT NEEDED
            
            % initialize results struct
            results = struct;
            results.T = [ 0 ];
            results.U = [ u0( end , : ) ];
            results.Y = [ y0( end , : ) ];
            results.K = [ 0 ];
            results.R = [ ref.y(1,:) ];
            results.X = [ y0( end , : ) ];
            results.Z = [ obj.lift.econ_full( zeta0' )' ]; % lifted states
            
            k = 1;
            while k < size( ref_sc , 1 )
                
                % current time
                t = k * obj.params.Ts;
                
                % get current state and input with delays
                if k == 1
                    current.y = obj.scaledown.y( y0 );   
                    current.u = obj.scaledown.u( u0 );  
                elseif k < nd + 1
                    y = [ y0( k : end-1 , : ) ; results.Y ];
                    u = [ u0( k : end-1 , : ) ; results.U ];
                    current.y = obj.scaledown.y( y );
                    current.u = obj.scaledown.u( u ); 
                else
                    y = results.Y( end - nd : end , : );
                    u = results.U( end - nd : end , : );
                    current.y = obj.scaledown.y( y ); 
                    current.u = obj.scaledown.u( u ); 
                end
                
                % isolate the reference trajectory over the horizon
                if k + Np <= size( ref_sc , 1 )
                    refhor = ref_sc( k : k + Np , :);
                else
                    refhor = ref_sc( k : end , : );     % repeat last entry
                end 
                
                % get optimal input over horizon
                [ U , z ] = obj.get_mpcInput( current , refhor );
                
                % if a solution was not found, break out of while loop
                if any( isnan(U) )
                    break;
                end
                
                % isolate input for this step (may need to make it U(1,:)
                u_kp1_sc = U( 2 , : );
                
                % scaleup the input for the results
                u_kp1 = obj.scaleup.u( u_kp1_sc )';
                
                % simulate the system over one time-step
                z_k = z;
                u_k_sc = obj.scaledown.u( results.U(end,:) );  % need to use previously calculated input NEED TO MAKE THIS CLEANER!!!
                z_kp1 = obj.model.A * z_k + obj.model.B * u_k_sc';
                x_kp1 = obj.model.C * z_kp1;
                y_kp1_sc = x_kp1;  % output juse scaled version of state since model was learned from observations
                y_kp1 = obj.scaleup.y( y_kp1_sc' )';  % scale output back up 
                
                % record updated results
                results.T = [ results.T ; t ];
                results.U = [ results.U ; u_kp1' ];
                results.Y = [ results.Y ; y_kp1' ];
                results.K = [ results.K ; k ];
                results.R = [ results.R ; obj.scaleup.y( ref_sc( k , : ) ) ];   % note that this is not scaled down
                results.X = [ results.X ; x_kp1' ];
                results.Z = [ results.Z ; z'  ]; % current lifted state
                
                k = k + 1;  % increment step counter
            end
        end
        
        %% bilinear MPC functions
        
        % get_costMatrices: Contructs the matrices for the mpc optim. problem
        function obj = get_costMatrices_bilinear( obj )
            %get_costMatrices_bilinear: Constructs cost the matrices for the mpc 
            % optimization problem.
            %   obj.cost has fields H, G, D, A, B, C, Q, R
            
            % define cost function matrices
            % Cost function is defined: U'HU + ( z0'G + Yr'D )U
            
            model = obj.model;
            
            % A
            N = size(model.A,1);
            A = sparse( N*(obj.horizon+1) , N );
            for i = 0 : obj.horizon
                A( (N*i + 1) : N*(i+1) , : ) = model.A^i ;
            end
            
            % B (mpc) zeros, just for size
            Bheight = N*(obj.horizon+1);
            Bwidth = obj.params.m * (obj.horizon);    % total columns in B matrix
            B = zeros( Bheight , Bwidth ); % initialze sparse B matrix
            
            % C: matrix that projects lifted state into reference trajectory space
            C = kron( speye(obj.horizon+1) , obj.projmtx);
            nproj = size( obj.projmtx , 1 );
            
            % Q: Error magnitude penalty
            Q = kron( speye(obj.horizon+1) , eye(nproj) * obj.cost_running); % error magnitude penalty (running cost) (default 0.1)
            Q(end-nproj+1 : end , end-nproj+1 : end) = eye(nproj) * obj.cost_terminal;    % (terminal cost) (default 100)
            
            % R: Input magnitude penalty
            R = kron( speye(obj.horizon) , eye(model.params.m) .* obj.cost_input );  % input magnitude penalty (for flaccy use 0.5e-2) (new videos used 0.5e-3)
            
            % set outputs
%             obj.cost.H = H; obj.cost.G = G; obj.cost.D = D; obj.cost.B = B; % constructed matrices
            obj.cost.A = A; obj.cost.C = C; obj.cost.Q = Q; obj.cost.R = R; % component matrices
            obj.cost.B = B; % all zeros, just included for size
            
            obj.cost.get_B = @obj.get_costB_bilinear;
            obj.cost.get_H = @obj.get_costH_bilinear;
            obj.cost.get_G = @obj.get_costG_bilinear;
            obj.cost.get_D = @obj.get_costD_bilinear;
        end
        
        % get_modelB_bilinear: Constructs the model state dependent B matrix
        function B = get_modelB_bilinear( obj , z )
%             zx = obj.lift.econ_full( x );
            zblock = kron( eye( obj.params.m ) , z );   % block matrix of zx
            B = obj.model.B * zblock;
        end
        
        % get_costB_bilinear: Constructs the mpc state dependent B matrix
        function B = get_costB_bilinear( obj , z )
            % z should be a row vector with 1 row or same number of rows as horizon
            N = size(obj.model.A,1);
            
            % B
            Bheight = N*(obj.horizon+1);
            Bcolwidth = obj.params.m;
            Bcol = zeros(Bheight , Bcolwidth);   % first column of B matrix

            for i = 1 : obj.horizon
                if size(z,1) > 1
                    Bmodel = obj.model.Beta( z(i,:)' );
                else
                    Bmodel = obj.model.Beta( z(1,:)' );
                end
                Bcol( (N*i + 1) : N*(i+1) , : ) = obj.model.A^(i-1) * Bmodel ;
            end
            
            Lshift = spdiags( ones( N*obj.horizon , 1 ) , -N , N*(obj.horizon+1) , N*(obj.horizon+1) );    % lower shift operator
            
            Bwidth = obj.params.m * (obj.horizon);    % total columns in B matrix
            Bblkwidth = obj.horizon;   % how many Bcol blocks wide B matrix is
            B = zeros( Bheight , Bwidth ); % initialze sparse B matrix
            B(: , 1:Bcolwidth) = Bcol;
            for i = 2 : Bblkwidth
                B(: , (i-1)*Bcolwidth+1 : i*Bcolwidth) = Lshift * B(: , (i-2)*Bcolwidth+1 : (i-1)*Bcolwidth);
            end
        end
        
        % get_costH_bilinear: Constructs the mpc state dependent H matrix
        function H = get_costH_bilinear( obj , z )
            B = obj.get_costB_bilinear( z );
            C = obj.cost.C;
            Q = obj.cost.Q;
            R = obj.cost.R;
            H = B' * C' * Q * C * B + R;
        end
        
        % get_costG_bilinear: Constructs the mpc state dependent G matrix
        function G = get_costG_bilinear( obj , z )
            B = obj.get_costB_bilinear( z );
            C = obj.cost.C;
            Q = obj.cost.Q;
            A = obj.cost.A;
            G = 2 * A' * C' * Q * C * B;
        end
        
        % get_costD_bilinear: Constructs the mpc state dependent D matrix
        function D = get_costD_bilinear( obj , z )
            B = obj.get_costB_bilinear( z );
            C = obj.cost.C;
            Q = obj.cost.Q;
            D = -2 * Q * C * B;
        end
        
        
        % get_constraintMatrices: Constructs the constraint matrices
        function obj = get_constraintMatrices_bilinear( obj )
            %get_constraintMatrices: Constructs the constraint matrices for
            % the mpc optimization problem.
            %   obj.constraints has fields L, M, F, E, (c?)
            %   F is for input constraints
            %   E is for state constraints
            
            % shorten some variable names
            Np = obj.horizon;     % steps in horizon
            params = obj.params;    % linear system model parameters
            cost = obj.cost;     % cost matrices
            
            F = []; E = [];     % initialize empty matrices
            c = [];
            
            % input_bounds
            if ~isempty( obj.input_bounds )
                num = 2*params.m;     % number of input bound constraints
                
                % F: input_bounds
                Fbounds_i = [ -speye(params.m) ; speye(params.m) ];    % diagonal element of F, for bounded inputs
                Fbounds = sparse( num * (Np+1) , size(cost.B,2) );  % no constraints, all zeros
                Fbounds( 1:num*Np , 1:Np*params.m ) = kron( speye(Np) , Fbounds_i );     % fill in nonzeros
                F = [ F ; Fbounds ];    % append matrix
                
                % E: input_bounds (just zeros)
                Ebounds = sparse( num * (Np+1) , size(cost.B,1) );  % no constraints, all zeros
                E = [ E ; Ebounds ];    % append matrix
                
                % c: input_bounds
                if isfield( obj.params , 'NLinput' )    % don't scale input if it's nonlinear
                    input_bounds_sc = obj.input_bounds;
                else
                    input_bounds_sc = obj.scaledown.u( obj.input_bounds' )';   % scaled down the input bounds
                end
                cbounds_i = [ -input_bounds_sc(:,1) ; input_bounds_sc(:,2) ]; % [ -umin ; umax ]
                cbounds = zeros( num * (Np+1) , 1);    % initialization
                cbounds(1 : num*Np) = kron( ones( Np , 1 ) , cbounds_i );     % fill in nonzeros
                c = [ c ; cbounds ];    % append vector
            end
            
            % input_slopeConst
            if ~isempty( obj.input_slopeConst )
                % F: input_slopeConst
                Fslope_i = speye(params.m);
                Fslope_neg = [ kron( speye(Np-1) , -Fslope_i ) , sparse( params.m * (Np-1) , params.m ) ];
                Fslope_pos = [ sparse( params.m * (Np-1) , params.m ) , kron( speye(Np-1) , Fslope_i ) ];
                Fslope_top = Fslope_neg + Fslope_pos;
                Fslope = [ Fslope_top ; -Fslope_top];
                F = [ F ; Fslope ];     % append matrix

                % E: input_slopeConst (just zeros)
                E = [ E ; sparse( 2 * params.m * (Np-1) , size(cost.B,1) ) ];
                
                % c: input_slopeConst
                if isfield( obj.params , 'NLinput' )    % don't scale slope if it's nonlinear
                    slope_lim = obj.input_slopeConst;
                else
                    slope_lim = obj.input_slopeConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                end
                cslope_top = slope_lim * ones( params.m * (Np-1) , 1 );
                cslope = [ cslope_top ; cslope_top ];
                c = [ c ; cslope ];     % append vector
            end
            
            % input_smoothConst
            if ~isempty( obj.input_smoothConst )
                % F: input_smoothConst
                Fsmooth_i = speye(params.m);
                Fsmooth_lI = [ kron( speye(Np-2) , Fsmooth_i ) , sparse( params.m * (Np-2) , 2 * params.m ) ];
                Fsmooth_2I = [ sparse( params.m * (Np-2) , params.m ) , kron( speye(Np-2) , -2*Fslope_i ) , sparse( params.m * (Np-2) , params.m ) ];
                Fsmooth_rI = [ sparse( params.m * (Np-2) , 2 * params.m ) , kron( speye(Np-2) , Fslope_i ) ];
                Fsmooth_top = Fsmooth_lI + Fsmooth_2I + Fsmooth_rI;
                Fsmooth = [ Fsmooth_top ; -Fsmooth_top ];
                F = [ F ; Fsmooth ];
                
                % E: input_smoothConst
                E = [ E ; sparse( 2 * params.m * (Np-2) , size(cost.B,1) ) ];
                
                % c: input_smoothConst
                smooth_lim = params.Ts^2 * obj.input_smoothConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                csmooth = smooth_lim * ones( size(Fsmooth,1) ,1);
                c = [ c ; csmooth ];
            end
            
            % state_bounds
            if ~isempty( obj.state_bounds )
                num = 2*params.n;   % number of state bound constraints
                
                % E: state_bounds
                Esbounds_i = [ -speye(params.n) ; speye(params.n) ];    % diagonal element of E, for bounding low dim. states (first n elements of lifted state)
                Esbounds = sparse( num * (Np+1) , size(cost.A,1) );  % no constraints, all zeros
                Esbounds( 1:num*(Np+1) , 1:(Np+1)*params.n ) = kron( speye(Np+1) , Esbounds_i );     % fill in nonzeros
                E = [ E ; Esbounds ];    % append matrix
                
                % F: state_bounds (all zeros)
                Fsbounds = zeros( size( Esbounds , 1 ) , size( cost.B , 2 ) );
                F = [ F ; Fsbounds ];    % append matrix
                
                % c: state_bounds
                state_bounds_sc = obj.scaledown.y( obj.state_bounds' )';    % scaled down state bounds
                csbounds_i = [ -state_bounds_sc(:,1) ; state_bounds_sc(:,2) ]; % [ -ymin ; ymax ]
                csbounds = kron( ones( Np+1 , 1 ) , csbounds_i );     % fill in nonzeros
                c = [ c ; csbounds ];    % append vector
            end
            
            % set outputs
            obj.constraints.F = F;
            obj.constraints.E = E;    
            obj.constraints.c = c;
            obj.constraints.get_L = @obj.get_constraintL_bilinear;
            obj.constraints.M = E * cost.A;
        end
        
        % get_constraintL_bilinear: Constructs the mpc state dependent L matrix
        function L = get_constraintL_bilinear( obj , z )
            B = obj.get_costB_bilinear( z );
            F = obj.constraints.F;
            E = obj.constraints.E;
            L = F + E * B;
        end
        
        
        % get_mpcInput_bilinear: Solve the mpc problem to get the input over entire horizon
        function [ U , z ]= get_mpcInput_bilinear( obj , traj , ref )
            %get_mpcInput: Soves the mpc problem to get the input over
            % entire horizon.
            %   traj - struct with fields y , u. Contains the measured
            %    states and inputs for the past ndelay+1 timesteps.
            %   ref - matrix containing the reference trajectory for the
            %    system over the horizon (one row per timestep).
            %   shape_bounds - [min_shape_parameters , max_shape_parameters] 
            %    This is only requred if system has shape constraints 
            %   (note: size is num of shape observables x 2)
            %   z - the lifted state at the current timestep
            
            % shorthand variable names
            Np = obj.horizon;       % steps in the horizon
            nd = obj.params.nd;     % number of delays
            
            % construct the current value of zeta
            [ ~ , zeta_temp ] = obj.get_zeta( traj );
            zeta = zeta_temp( end , : )';   % want most recent points
            
            % lift zeta
            if obj.loaded
                z = obj.lift.econ_full_loaded( zeta , traj.what(end,:)' );
            else
                z = obj.lift.econ_full( zeta );
            end
            zrow = z';
            
            % check that reference trajectory has correct dimensions
            if size( ref , 2 ) ~= size( obj.projmtx , 1 )
                error('Reference trajectory is not the correct dimension');
            elseif size( ref , 1 ) > Np + 1
                ref = ref( 1 : Np + 1 , : );    % remove points over horizon
            elseif size( ref , 1 ) < Np + 1
                ref_temp = kron( ones( Np+1 , 1 ) , ref(end,:) );
                ref_temp( 1 : size(ref,1) , : ) = ref;
                ref = ref_temp;     % repeat last point for remainer of horizon
            end
            
            % vectorize the reference trajectory
            Yr = reshape( ref' , [ ( Np + 1 ) * size(ref,2) , 1 ] );
            
            % setup matrices for gurobi solver
            H = obj.get_costH_bilinear( zrow );
            G = obj.get_costG_bilinear( zrow );
            D = obj.get_costD_bilinear( zrow );
            f = ( z' * G + Yr' * D )';
            A = obj.get_constraintL_bilinear( zrow );
            b = - obj.constraints.M * z + obj.constraints.c;
            
            % tack on "memory" constraint to fix initial input u_0
            Atack = [ [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - obj.params.m ) ];
%             Atack_bot = [ sparse( 2*obj.params.m , obj.params.m) , [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - 2*obj.params.m ) ];
%             Atack = [ Atack_top ; Atack_bot ];
            btack = [ traj.u(end,:)' ; -traj.u(end,:)' ];
            A = [A ; Atack];    % tack on memory constraint
            b = [b ; btack];
            
            % solve the MPC problem
%             Uvec = quadprog_gurobi( H , f , A , b );   % solve using gurobi (returns NaNs of cannot be solved)
            Uvec = quadprog( 2*H , f , A , b );     % solve using matlab
            
            % reshape the output so each input will have one row (first row equals current input)
            U = reshape( Uvec , [ obj.params.m , Np ] )';
        end
        
        % get_mpcInput_bilinear_iter: Solve the mpc problem to get the input over entire horizon
        function [ U , z ]= get_mpcInput_bilinear_iter( obj , traj , ref , iter)
            %get_mpcInput_bilinear_iter: Soves the mpc problem to get the 
            % input over entire horizon.
            %   traj - struct with fields y , u. Contains the measured
            %    states and inputs for the past ndelay+1 timesteps.
            %   ref - matrix containing the reference trajectory for the
            %    system over the horizon (one row per timestep).
            %   shape_bounds - [min_shape_parameters , max_shape_parameters] 
            %    This is only requred if system has shape constraints 
            %   (note: size is num of shape observables x 2)
            %   z - the lifted state at the current timestep
            %   iter - number of iterations of the problem to perform
            
            % shorthand variable names
            Np = obj.horizon;       % steps in the horizon
            nd = obj.params.nd;     % number of delays
            
            % construct the current value of zeta
            [ ~ , zeta_temp ] = obj.get_zeta( traj );
            zeta = zeta_temp( end , : )';   % want most recent points
            
            % lift zeta
            if obj.loaded
                z = obj.lift.econ_full_loaded( zeta , traj.what(end,:)' );
            else
                z = obj.lift.econ_full( zeta );
            end
            zrow = z';
            
            % check that reference trajectory has correct dimensions
            if size( ref , 2 ) ~= size( obj.projmtx , 1 )
                error('Reference trajectory is not the correct dimension');
            elseif size( ref , 1 ) > Np + 1
                ref = ref( 1 : Np + 1 , : );    % remove points over horizon
            elseif size( ref , 1 ) < Np + 1
                ref_temp = kron( ones( Np+1 , 1 ) , ref(end,:) );
                ref_temp( 1 : size(ref,1) , : ) = ref;
                ref = ref_temp;     % repeat last point for remainer of horizon
            end
            
            % vectorize the reference trajectory
            Yr = reshape( ref' , [ ( Np + 1 ) * size(ref,2) , 1 ] );
            
            % define linear inequality constraints
            A = obj.get_constraintL_bilinear( zrow );
            b = - obj.constraints.M * z + obj.constraints.c;
            
            % tack on "memory" constraint to fix initial input u_0
            Atack = [ [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - obj.params.m ) ];
%             Atack_bot = [ sparse( 2*obj.params.m , obj.params.m) , [ speye( obj.params.m ) ; -speye( obj.params.m ) ] , sparse( 2*obj.params.m , size(A,2) - 2*obj.params.m ) ];
%             Atack = [ Atack_top ; Atack_bot ];
            btack = [ traj.u(end,:)' ; -traj.u(end,:)' ];
            A = [A ; Atack];    % tack on memory constraint
            b = [b ; btack];
            
            zhorizon = zrow;    % initial value
%             zhorizon = ( obj.model.A * zrow' + obj.get_modelB_bilinear( zrow' ) * traj.u(end,:)' )';    % second z value
            for i = 1 : iter
                % setup matrices for QP solver
                H = obj.get_costH_bilinear( zhorizon );
                G = obj.get_costG_bilinear( zhorizon );
                D = obj.get_costD_bilinear( zhorizon );
                f = ( zrow * G + Yr' * D )';
                
                % solve the MPC problem
%                 Uvec = quadprog_gurobi( H , f , A , b );   % solve using gurobi (returns NaNs of cannot be solved)
                Uvec = quadprog( 2*H , f , A , b );     % solve using matlab
                uhorizon = reshape( Uvec , [ obj.params.m , Np ] )';
                
                if i == iter    % don't execut remaininc code in loop if last iteration
                    break;
                end
                
                % solve for lifted state sequence corresponding to those inputs
                zhorizon = zeros( Np+1 , obj.params.N );
                zhorizon(1,:) = zrow;
                for j = 1 : Np
                    zhorizon(j+1,:) = ( obj.model.A * zhorizon(j,:)' + obj.get_modelB_bilinear( zhorizon(j,:)' ) * uhorizon(j,:)' )';
                end
%                 % first version...
%                 Zvec = obj.cost.A * z + obj.get_costB_bilinear( zhorizon ) * Uvec;
%                 zhorizon = reshape( Zvec , [ obj.params.N , Np+1 ] )';
            end
            
            % reshape the output so each input will have one row (first row equals current input)
%             U = reshape( Uvec , [ obj.params.m , Np ] )';
            U = uhorizon;
        end
        
        %% nonlinear MPC functions

        % get_costMatrices_nonlinear: Contructs the matrices for the mpc optim. problem
        function obj = get_costMatrices_nonlinear( obj )
            %get_costMatrices: Constructs cost the matrices for the mpc 
            % optimization problem.
            %   obj.cost has fields H, G, D, A, B, C, Q, R
            
            % define cost function matrices
            % Cost function is defined: U'HU + ( z0'G + Yr'D )U
            
            % Selection matrices
            Ny = obj.params.n*(obj.horizon+1); % dimension of vectorized sequence of lifted states
            Nu = obj.params.m*obj.horizon; % dimension of vectorized sequence of inputs
            Sy = [ eye( Ny ) , zeros( Ny , Nu ) ];  % isolates lifted states from decision variable
            Su = [ zeros( Nu , Ny ) , eye( Nu ) ];  % isolates inputs from decision variable
               
            % C: matrix that projects state into reference trajectory space
            C = kron( speye(obj.horizon+1) , obj.projmtx);
            nproj = size( obj.projmtx , 1 );
            
            % Q: Error magnitude penalty
            Q = kron( speye(obj.horizon+1) , eye(nproj) * obj.cost_running); % error magnitude penalty (running cost) (default 0.1)
            Q(end-nproj+1 : end , end-nproj+1 : end) = eye(nproj) * obj.cost_terminal;    % (terminal cost) (default 100)
            
            % R: Input magnitude penalty
            R = kron( speye(obj.horizon) , eye(obj.model.params.m) .* obj.cost_input );  % input magnitude penalty (for flaccy use 0.5e-2) (new videos used 0.5e-3)

            % H, G, D
            H = Sy' * C' * Q * C * Sy + Su' * R * Su;
            D = -2 * Q * C * Sy;
            
            % set outputs
            obj.cost.H = H; obj.cost.D = D; % constructed matrices
            obj.cost.C = C; obj.cost.Q = Q; obj.cost.R = R; % component matrices
            obj.params.Sy = Sy; obj.params.Su = Su; % selection matrices
            obj.params.Ny = Ny; obj.params.Nu = Nu; % dimension of decision variables
        end
        
        % get_constraintMatrices_nonlinear: Constructs the constraint matrices
        function obj = get_constraintMatrices_nonlinear( obj )
            %get_constraintMatrices_nonlinear: Constructs the constraint matrices for
            % the mpc optimization problem.
            %   obj.constraints has fields A, F, E, c
            %   F is for input constraints
            %   E is for state constraints
            
            % shorten some variable names
            Np = obj.horizon;     % steps in horizon
            params = obj.params;    % linear system model parameters
            cost = obj.cost;     % cost matrices
            
            F = []; E = [];     % initialize empty matrices
            c = [];
            
            % input_bounds
            if ~isempty( obj.input_bounds )
                num = 2*params.m;     % number of input bound constraints
                
                % F: input_bounds
                Fbounds_i = [ -speye(params.m) ; speye(params.m) ];    % diagonal element of F, for bounded inputs
                Fbounds = sparse( num * (Np+0) , params.Nu );  % no constraints, all zeros
                Fbounds( 1:num*Np , 1:Np*params.m ) = kron( speye(Np) , Fbounds_i );     % fill in nonzeros
                F = [ F ; Fbounds ];    % append matrix
                
                % E: input_bounds (just zeros)
                Ebounds = sparse( num * (Np+0) , params.Ny );  % no constraints, all zeros
                E = [ E ; Ebounds ];    % append matrix
                
                % c: input_bounds
                input_bounds_sc = obj.scaledown.u( obj.input_bounds' )';   % scale down the input bounds
                cbounds_i = [ -input_bounds_sc(:,1) ; input_bounds_sc(:,2) ]; % [ -umin ; umax ]
                cbounds = zeros( num * (Np+0) , 1);    % initialization
                cbounds(1 : num*Np) = kron( ones( Np , 1 ) , cbounds_i );     % fill in nonzeros
                c = [ c ; cbounds ];    % append vector
            end
            
            % input_slopeConst
            if ~isempty( obj.input_slopeConst )
                % F: input_slopeConst
                Fslope_i = speye(params.m);
                Fslope_neg = [ kron( speye(Np-1) , -Fslope_i ) , sparse( params.m * (Np-1) , params.m ) ];
                Fslope_pos = [ sparse( params.m * (Np-1) , params.m ) , kron( speye(Np-1) , Fslope_i ) ];
                Fslope_top = Fslope_neg + Fslope_pos;
                Fslope = [ Fslope_top ; -Fslope_top];
                F = [ F ; Fslope ];     % append matrix

                % E: input_slopeConst (just zeros)
                E = [ E ; sparse( 2 * params.m * (Np-1) , params.Ny ) ];
                
                % c: input_slopeConst
                slope_lim = obj.input_slopeConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                cslope_top = slope_lim * ones( params.m * (Np-1) , 1 );
                cslope = [ cslope_top ; cslope_top ];
                c = [ c ; cslope ];     % append vector
            end
            
            % input_smoothConst
            if ~isempty( obj.input_smoothConst )
                % F: input_smoothConst
                Fsmooth_i = speye(params.m);
                Fsmooth_lI = [ kron( speye(Np-2) , Fsmooth_i ) , sparse( params.m * (Np-2) , 2 * params.m ) ];
                Fsmooth_2I = [ sparse( params.m * (Np-2) , params.m ) , kron( speye(Np-2) , -2*Fslope_i ) , sparse( params.m * (Np-2) , params.m ) ];
                Fsmooth_rI = [ sparse( params.m * (Np-2) , 2 * params.m ) , kron( speye(Np-2) , Fslope_i ) ];
                Fsmooth_top = Fsmooth_lI + Fsmooth_2I + Fsmooth_rI;
                Fsmooth = [ Fsmooth_top ; -Fsmooth_top ];
                F = [ F ; Fsmooth ];
                
                % E: input_smoothConst
                E = [ E ; sparse( 2 * params.m * (Np-2) , params.Ny ) ];
                
                % c: input_smoothConst
                smooth_lim = params.Ts^2 * obj.input_smoothConst * mean( params.scale.u_factor );  % scale down the 2nd deriv. limit
                csmooth = smooth_lim * ones( size(Fsmooth,1) ,1);
                c = [ c ; csmooth ];
            end
            
            % state_bounds
            if ~isempty( obj.state_bounds )
                num = 2*params.n;   % number of state bound constraints
                
                % E: state_bounds
                Esbounds_i = [ -speye(params.n) ; speye(params.n) ];    % diagonal element of E, for bounding low dim. states (first n elements of lifted state)
                Esbounds = sparse( num * (Np+1) , params.Ny );  % no constraints, all zeros
                Esbounds( 1:num*(Np+1) , 1:(Np+1)*params.n ) = kron( speye(Np+1) , Esbounds_i );     % fill in nonzeros
                E = [ E ; Esbounds ];    % append matrix
                
                % F: state_bounds (all zeros)
                Fsbounds = zeros( size( Esbounds , 1 ) , params.Nu );
                F = [ F ; Fsbounds ];    % append matrix
                
                % c: state_bounds
                state_bounds_sc = obj.scaledown.y( obj.state_bounds' )';    % scaled down state bounds
                csbounds_i = [ -state_bounds_sc(:,1) ; state_bounds_sc(:,2) ]; % [ -ymin ; ymax ]
                csbounds = kron( ones( Np+1 , 1 ) , csbounds_i );     % fill in nonzeros
                c = [ c ; csbounds ];    % append vector
            end
            
            % create symbolic functions for gradients
            obj.model.dFdzeta_sym = jacobian( obj.model.F_sym , obj.params.zeta );
            obj.model.dFdu_sym = jacobian( obj.model.F_sym , obj.params.u );
            obj.model.dFdzeta_func = matlabFunction( obj.model.dFdzeta_sym , 'Vars' , {obj.params.zeta, obj.params.u} );
            obj.model.dFdu_func = matlabFunction( obj.model.dFdu_sym , 'Vars' , {obj.params.zeta, obj.params.u} );
            
            % set outputs
            obj.constraints.F = F;
            obj.constraints.E = E;    
            obj.constraints.c = c;
            if ~isempty(E)
                obj.constraints.A = E * obj.params.Sy + F * obj.params.Su;
            else
                obj.constraints.A = [];
            end
        end
        
        % cost_nmpc: Evaluates the cost function for NMPC
        function [ cost , grad ] = cost_nmpc( obj , X , Yr )
           %cost_nmpc: Evaluates NMPC cost function
           %    X = [Z;U] - NMPC decision variable where Z is vectorized
           %    sequence of lifted states and U is vectorized sequence of
           %    inputs.
           %    Yr - vectorized sequence of desired outputs
           
           cost = X' * obj.cost.H * X + Yr' * obj.cost.D * X;  % cost function
           grad = 2 * obj.cost.H * X + obj.cost.D' * Yr;    % gradient of cost function
        end
        
        % nonlcon_nmpc: Nonlinear constraints for NMPC
        function [ c , ceq , gc , gceq] = nonlcon_nmpc( obj , X )
            %nonlcon_nmpc: Nonlinear constraints for NMPC
            % c - inequality constraints c <= 0
            % ceq - equality constraints c = 0
            % gc - gradient of inequality constraints
            % gceq - gradient of equality constraints
            
            Y = obj.params.Sy * X;  % vectorized lifted states
            U = obj.params.Su * X;  % vecotrized inputs
            
            % set equality constraints
            ceq = zeros( size(Y,1)-obj.params.n , 1 ); % preallocate
            gceq = zeros( size(Y,1)-obj.params.n , size(X,1) );
            for i = 1 : obj.horizon
                y_ind = (i-1)*obj.params.n+1 : i*obj.params.n;
                u_ind = (i-1)*obj.params.m+1 : i*obj.params.m;
                yk = Y(y_ind);
                uk = U(u_ind);
                ceq(y_ind) = Y( y_ind + obj.params.n ) - obj.model.F_func( yk , uk );
                
                % specify gradients
                gceq( y_ind , y_ind) = -obj.model.dFdzeta_func( yk , uk );
                gceq( y_ind , y_ind+obj.params.n ) = eye( obj.params.n );
                gceq( y_ind , obj.params.Ny + u_ind ) = -obj.model.dFdu_func( yk , uk );
            end
            gceq = gceq'; % matlab wants columns to correspond to constraints
            
            % inequality constraints are equality constraints with tolerance
%             tol = 1e-2;
%             c = [ ceq - tol ; -ceq - tol ];
%             gc = [ gceq , -gceq ];
%             ceq = [];   % negate equality constraints
%             gceq = [];
            
            % no inequality constraints 
            c = 0;
            gc = zeros( 1 , size(X,1) )';
        end
        
        % get_mpcInput_nonlinear: Solve the mpc problem to get the input over entire horizon
        function [ U , z ]= get_mpcInput_nonlinear( obj , traj , ref )
            %get_mpcInput: Soves the mpc problem to get the input over
            % entire horizon.
            %   traj - struct with fields y , u. Contains the measured
            %    states and inputs for the past ndelay+1 timesteps.
            %   ref - matrix containing the reference trajectory for the
            %    system over the horizon (one row per timestep).
            %   shape_bounds - [min_shape_parameters , max_shape_parameters] 
            %    This is only requred if system has shape constraints 
            %   (note: size is num of shape observables x 2)
            %   z - the lifted state at the current timestep
            
            % shorthand variable names
            Np = obj.horizon;       % steps in the horizon
            nd = obj.params.nd;     % number of delays
            
            % construct the current value of zeta
            [ ~ , zeta_temp ] = obj.get_zeta( traj );
            zeta = zeta_temp( end , : )';   % want most recent points
            
            % check that reference trajectory has correct dimensions
            if size( ref , 2 ) ~= size( obj.projmtx , 1 )
                error('Reference trajectory is not the correct dimension');
            elseif size( ref , 1 ) > Np + 1
                ref = ref( 1 : Np + 1 , : );    % remove points over horizon
            elseif size( ref , 1 ) < Np + 1
                ref_temp = kron( ones( Np+1 , 1 ) , ref(end,:) );
                ref_temp( 1 : size(ref,1) , : ) = ref;
                ref = ref_temp;     % repeat last point for remainer of horizon
            end
            
            % vectorize the reference trajectory
            Yr = reshape( ref' , [ ( Np + 1 ) * size(ref,2) , 1 ] );
            
            % setup matrices for fmincon
            A = obj.constraints.A;
            b = obj.constraints.c;
            
            % fix the initial state and input
            Aeq = [ eye(obj.params.n) , zeros( obj.params.n , obj.params.Ny - obj.params.n + obj.params.Nu ) ;...
                    zeros( obj.params.m , obj.params.Ny  ) , eye( obj.params.m ) , zeros( obj.params.m , obj.params.Nu - obj.params.m )];
            beq = [ zeta ; traj.u(end,:)' ]; 
            
            % set initial condition for fmincon (repeat of current state and input)
            X0 = [ kron( ones(Np+1,1) , zeta ) ; kron( ones(Np,1) , traj.u(end,:)' ) ];
%             X0 = zeros( obj.params.Ny + obj.params.Nu , 1 );    % zero IC
%             X0 = [ Yr ; kron( ones(Np,1) , traj.u(end,:)' ) ];  % from ref traj
%             X0 = 2*rand( obj.params.Ny+obj.params.Nu , 1 ) - 1; % random
%             X0 = [ Yr ; Yr(2:end) ];  % known optimal solution
            
            % solve the MPC problem
            fun = @(X) obj.cost_nmpc(X,Yr);
            nonlcon = @(X) obj.nonlcon_nmpc(X);
            options = optimoptions( 'fmincon' ,...
                                    'Algorithm' , 'sqp' ,...
                                    'SpecifyObjectiveGradient' , true ,...
                                    'SpecifyConstraintGradient' , true ,...
                                    'Display' , 'iter' , ...
                                    'ConstraintTolerance' , 1e-6 , ...
                                    'ScaleProblem', false );
            X = fmincon(fun,X0,A,b,Aeq,beq,[],[],nonlcon,options);     % solve using fmincon
            Yvec = X( 1 : obj.params.n * (Np+1) );   % vectorized outputs over horizon
            Uvec = X( obj.params.n * (Np+1) + 1 : end ); % vectorize inputs over horizon
            
            % reshape the output so each input will have one row (first row equals current input)
            U = reshape( Uvec , [ obj.params.m , Np ] )';
            z = [ zeta ; zeros( obj.params.N - obj.params.n , 1 )]; % current lifted state isn't actually lifted for this system
        end
        
        %% nlmpc matlab object (UNUSED: DELETE WHEN READY)
        % get_nlmpc_controller: Construct nlmpc Matlab object
        function obj = get_nlmpc_controller(obj)
           
            % crete nlmpc object (requires MPC and Optimization Toolboxes)
            nlobj = nlmpc( obj.params.n , size(obj.projmtx,1) , obj.params.m );
            
            % change default nlmpc object parameters
            nlobj.Ts = obj.params.Ts;
            nlobj.PredictionHorizon = obj.horizon;
            nlobj.Model.StateFcn = @(zeta,u) obj.model.F_func(zeta,u);
            nlobj.Model.OutputFcn = matlabFunction( obj.projmtx * obj.params.zeta , 'Vars' , {obj.params.zeta , obj.params.u} );
%             nlobj.Model.OutputFcn = @(zeta,u) obj.model.C * obj.params.zeta;
            nlobj.Model.IsContinuousTime = false;
%             nlobj.OutputVariables.Min = obj.input_bounds(1,1);
%             nlobj.OutputVariables.Max = obj.input_bounds(1,2);
            
            % specify cost function
            nlobj.Optimization.CustomCostFcn = @obj.nlmpc_cost_function;
            nlobj.Optimization.CustomEqConFcn = @obj.nlmpc_eqcon_function;
            nlobj.Optimization.CustomIneqConFcn = @obj.nlmpc_ineqcon_function;
            
            % specify jacobians
            nlobj.Jacobian.StateFcn = @(zeta,u) jacobian( obj.model.F_sym , obj.params.zeta );
            nlobj.Jacobian.OutputFcn = @(zeta,u) jacobian( obj.model.C * obj.params.zeta , obj.params.zeta );
%             nlobj.Jacobian.CustomCostFcn = @(zeta,u) obj.nlmpc_cost_jacobian;
%             nlobj.Jacobian.CustomEqConFcn = @(zeta,u) obj.nlmpc_eqcon_jacobian;
%             nlobj.Jacobian.CustomIneqConFcn = @(zeta,u) obj.nlmpc_ineqcon_jacobian;
            
            % specify output
            obj.nlmpc_controller = nlobj;
        end
        
        % nlmpc_cost_function: Cost function needed for Matlab nlmpc object
        function cost = nlmpc_cost_function( obj , Z , U , e , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            ref = [ data.References ; data.References(end,:) ];   % needs additional row
            Yr = reshape( ref' , [ numel(ref) , 1 ] );
            
            % calculate cost
            cost = obj.cost_nmpc( X , Yr );
        end
        
        % nlmpc_cost_jacobian: Cost function needed for Matlab nlmpc object
        function cost_jacobian = nlmpc_cost_jacobian( obj , Z , U , e , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            ref = [ data.References ; data.References(end,:) ];   % needs additional row
            Yr = reshape( ref' , [ numel(ref) , 1 ] );
            
            % calculate cost
            [ cost , gradcost ] = obj.cost_nmpc( X , Yr );
            cost_jacobian = gradcost( 1 : length(Zvec) , : )';
        end
        
        % nlmpc_eqcon_function: Equality constraints needed for Matlab nlmpc object
        function ceq = nlmpc_eqcon_function( obj , Z , U , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            
            % calculate equality constraints
            [ c , ceq , gc , gceq] = obj.nonlcon_nmpc( X );
        end
        
        % nlmpc_eqcon_jacobian: Equality constraints needed for Matlab nlmpc object
        function ceq_jacobian = nlmpc_eqcon_jacobian( obj , Z , U , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            
            % calculate equality constraints
            [ c , ceq , gc , gceq] = obj.nonlcon_nmpc( X );
            ceq_jacobian = gceq( 1 : length(Zvec) , : )';
        end
        
        % nlmpc_ineqcon_function: Inequality constraints needed for Matlab nlmpc object
        function c = nlmpc_ineqcon_function( obj , Z , U , e , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            
            % calculate equality constraints
            [ c , ceq , gc , gceq] = obj.nonlcon_nmpc( X );
        end
        
        % nlmpc_ineqcon_jacobian: Inequality constraints needed for Matlab nlmpc object
        function c_jacobian = nlmpc_ineqcon_jacobian( obj , Z , U , e , data)
            % vectorize inputs
            Zvec = reshape( Z' , [ numel(Z) , 1 ] );
            Uvec = reshape( U' , [ numel(U) , 1 ] );
            X = [ Zvec ; Uvec( 1 : end - obj.params.m ) ];  % remove final value
            
            % calculate equality constraints
            [ c , ceq , gc , gceq] = obj.nonlcon_nmpc( X );
            c_jacobian = gc( 1 : length(Zvec) , : )';
        end
        
        % get_nlmpc_input
        function unext = get_nlmpc_input( obj , traj , ref )
            % use matlab nonlinear mpc object
            unext = nlmpcmove( obj.nlmpc_controller , traj.y(end,:) , traj.u(end,:) ,ref );
        end
        
        %% load estimation
    
        % estimate_load_linear (infer the load based on dynamics)
        function [ what , resnorm ] = estimate_load_linear( obj , ypast , upast , whatpast )
            % estimate_load_linear: Estimate the load given measurements over a 
            % past horizon.
            %   ypast - [hor x n], output measurements over previous hor steps
            %   upast - [hor x m], inputs over previous hor steps
            %   whatpast - [1 x nw], load estimate at previous step (optional)
            %   resnorm - squared 2-norm of residual of cost function
            % Note: This doesn't work for delays yet...
            %       This doesn't work for bilinear systems yet
            
            hor_y = size( ypast , 1 ); % length of past horizon
            if size(upast,1) ~= hor_y
                error('Input arguments must have the same number of rows');
            end
            
            % construct zeta from input arguments
            traj.y = ypast; traj.u = upast;
            [ ~ , zetapast ] = obj.get_zeta( traj );
            hor = size( zetapast , 1 ); % length of past horizon with delays
            
            % stack Omega and u vertically
            Omega = zeros( obj.params.N * (obj.params.nw+1) * (hor-1) , obj.params.nw+1 );
            for i = 1 : hor-1
                gy_i = obj.lift.econ_full( zetapast(i,:)' );    % should be zeta, but okay with no delays
                Omega_i = kron( eye(obj.params.nw+1) , gy_i );
                ind1 = (obj.params.N*(obj.params.nw+1))*(i-1)+1;
                ind2 = (obj.params.N*(obj.params.nw+1))*i;
                Omega( ind1 : ind2 , : ) = Omega_i; 
            end
            U = reshape( upast(obj.params.nd+1:end-1,:)' , [ obj.params.m * (hor-1) , 1 ] );
            Zeta = reshape( zetapast( 2:end , 1:obj.params.nzeta )' , [ obj.params.nzeta * (hor-1) , 1 ] );
            
            % cost function matrices
            CAstack = kron(eye(hor-1) , obj.model.A( 1:obj.params.nzeta , : ) );    % accounts for delays
            CBstack = kron(eye(hor-1) , obj.model.B( 1:obj.params.nzeta , : ) );    % accounts for delays
            Clsqlin = CAstack * Omega;
            dlsqlin = Zeta - CBstack * U;
            
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
%             Aeq = blkdiag( 1 , zeros(obj.params.nw , obj.params.nw) );
            Aeq = blkdiag( 1 , 0 , 1 ); % DEBUG: ensure last element is zero and first element is one
            beq = [ 1 ; zeros(obj.params.nw,1) ]; % ensure first elements is 1
            lb = -ones(obj.params.nw+1,1);  % load should be in [-1,1]
            ub = ones(obj.params.nw+1,1);   % load should be in [-1,1]
            
            % solve for what
            [ sol , resnorm ] = lsqlin( Clsqlin , dlsqlin , A , b , Aeq , beq , lb , ub );  % solve for what using constrained least squares solver
            what = sol(2:end);
        end
        
        % estimate_load_bilinear (infer the load based on dynamics)
        function [ what , resnorm ] = estimate_load_bilinear( obj , ypast , upast , whatpast )
            % estimate_load_bilinear: Estimate the load given measurements over a 
            % past horizon.
            %   ypast - [hor x n], output measurements over previous hor steps
            %   upast - [hor x m], inputs over previous hor steps
            %   whatpast - [1 x nw], load estimate at previous step (optional)
            %   resnorm - squared 2-norm of residual of cost function
            % Note: This doesn't work for delays yet...
            %       This doesn't work for bilinear systems yet
            
            hor_y = size( ypast , 1 ); % length of past horizon
            if size(upast,1) ~= hor_y
                error('Input arguments must have the same number of rows');
            end
            
            % construct zeta from input arguments
            traj.y = ypast; traj.u = upast;
            [ ~ , zetapast ] = obj.get_zeta( traj );
            hor = size( zetapast , 1 ); % length of past horizon with delays
            
            % construct the RHS regression matrix ( LHS = RHS * [1;w] )
            RHS = zeros( obj.params.nzeta * (hor-1) , obj.params.nw+1 );
%             RHS = zeros( obj.params.nzeta * obj.params.m * (hor-1) , obj.params.nw+1 );
            for i = 1 : hor-1
                gy_i = obj.lift.econ_full( zetapast(i,:)' );   
                Omega_i = kron( eye(obj.params.nw+1) , gy_i );
                 
                RHS_i_CB = zeros( obj.params.nzeta , obj.params.nw+1 );
                for j = 1 : obj.params.m
                    CA = obj.model.A( 1:obj.params.nzeta , : );
                    B_col_range = (j-1)*obj.params.N*(obj.params.nw+1)+1 : j*obj.params.N*(obj.params.nw+1);
                    CB = obj.model.B( 1:obj.params.nzeta , B_col_range );
                    RHS_i_CB = RHS_i_CB + CB * Omega_i * upast(i,j);
                end
                RHS_i = CA * Omega_i + RHS_i_CB;
                ind1 = obj.params.nzeta * (i-1) + 1;
                ind2 = obj.params.nzeta * i;
                RHS( ind1:ind2 , :) = RHS_i;
               
%                 RHS_i = zeros( obj.params.nzeta * obj.params.m , obj.params.nw+1 );
%                 for j = 1 : obj.params.m
%                     CA = obj.model.A( 1:obj.params.nzeta , : );
%                     B_col_range = (j-1)*obj.params.N*(obj.params.nw+1)+1 : j*obj.params.N*(obj.params.nw+1);
%                     CB = obj.model.B( 1:obj.params.nzeta , B_col_range );
%                     RHS_ij = CA * Omega_i + CB * Omega_i * upast(i,j);
%                     ind1 = (j-1)*obj.params.nzeta + 1;
%                     ind2 = j*obj.params.nzeta;
%                     RHS_i( ind1:ind2 , : ) = RHS_ij; 
%                 end
%                 ind1 = (obj.params.nzeta * obj.params.m) * (i-1) + 1;
%                 ind2 = (obj.params.nzeta * obj.params.m) * i;
%                 RHS( ind1:ind2 , :) = RHS_i;
            end
            
            % construct the LHS regression matrix ( LHS = RHS * [1;w] )
            LHS = reshape( zetapast( 2:end , 1:obj.params.nzeta )' , [ obj.params.nzeta * (hor-1) , 1 ] );
%             zetapast_rep = repelem( zetapast , obj.params.m , 1 );  % duplicate each row of zetapast m times
%             LHS = reshape( zetapast_rep( obj.params.m+1:end , 1:obj.params.nzeta )' , [ obj.params.nzeta * (hor-1) * obj.params.m , 1 ] ); 
                 
            % cost function matrices
            Clsqlin = RHS;
            dlsqlin = LHS;
            
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
%             Aeq = blkdiag( 1 , 0 , 1 ); % DEBUG: ensure last element is zero and first element is one
%             Aeq = blkdiag( 1 , 1 , 0 ); % DEBUG: ensure second element is zero and first element is one
            beq = [ 1 ; zeros(obj.params.nw,1) ]; % ensure first elements is 1
            lb = -ones(obj.params.nw+1,1);  % load should be in [-1,1]
            ub = ones(obj.params.nw+1,1);   % load should be in [-1,1]
            
            % solve for what
            [ sol , resnorm ] = lsqlin( Clsqlin , dlsqlin , A , b , Aeq , beq , lb , ub );  % solve for what using constrained least squares solver
            what = sol(2:end);
        end
    
    end
end


















