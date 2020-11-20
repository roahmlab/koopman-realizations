classdef Arm
    %arm: Class with system properties and equations of motion inside
    %   This is for a planar manipulator arm specifically.
    
    properties
        params struct;  % model parameters
        fcns struct;   % equation of motion functions
        output_type;  % choice for the system output. Either 'angles' , 'markers' , or 'endeff' 
    end
    
    methods
        function obj = Arm( params , varargin )
            %Construct an instance of EOM class
            %   Detailed explanation goes here
            obj.params = params;
            obj.fcns = obj.set_EOM;
            
            % set default value of the output type ('angles' or 'markers')
            output_type = 'angles';
            
            % replace default values with user input values
            obj = obj.parse_args( varargin{:} );
        end
        
        % parse_args: Parses the Name, Value pairs in varargin
        function obj = parse_args( obj , varargin )
            %parse_args: Parses the Name, Value pairs in varargin of the
            % constructor, and assigns property values
            for idx = 1:2:length(varargin)
                obj.(varargin{idx}) = varargin{idx+1};
            end
        end
        
        %% transformations
        
        % alpha2theta
        function theta = alpha2theta( obj , alpha )
            %alpha2theta: Converts relative joint angles (alpha) to absolute joint angles (theta).
            
            theta = zeros( size(alpha) );
            
            % if input is symbolic, so should output
            if isa( alpha , 'sym' )
                theta = sym(theta);
            end
            
            for i = 1 : length(alpha)
                theta(i) =  sum(alpha(1:i));
            end
        end
        
        % alpha2xvec (gives x vectorized)
        function [ x , x_cm ] = alpha2xvec( obj , alpha )
            %alpha2xvec: Converts relative joint angles (alpha) to coordinates of joints (x)
            %   and the coordinates of the links' centers of mass (x_cm).
            %   x = [ x_0 ; y_0 ; x_1 ; y_1 ; ... ]
            
            x = zeros( ( obj.params.Nlinks + 1 ) * 2 ,  1 );
            x_cm = zeros( obj.params.Nlinks * 2 , 1 );
            
            % if input is symbolic, so should output
            if isa( alpha , 'sym' )
                x = sym(x);
                x_cm = sym(x_cm);
            end
            
            % convert to absolute joint angles (wrt vertical)
            theta = obj.alpha2theta(alpha);
            
            % convert to coordinates of each joint (note there is 1 more joint than link)
            for i = 1 : length(alpha)
                xim1 = x(2*(i-1)+1 : 2*i, 1);
                x_cm(2*(i-1)+1 : 2*i, 1) = xim1 + obj.params.l/2 * [ -sin( theta(i) ) ; cos( theta(i) ) ];
                x(2*i+1 : 2*(i+1), 1) = xim1 + obj.params.l * [ -sin( theta(i) ) ; cos( theta(i) ) ];
            end
        end
        
        % alpha2x (gives x where rows are x,y coordinate pairs)
        function [ x , xcm ] = alpha2x( obj , alpha )
            % alpha2x: (gives x where rows are x,y coordinate pairs)
            [ x_vec ,xcm_vec ] = obj.alpha2xvec( alpha );
            x = reshape( x_vec , [ 2 , obj.params.Nlinks+1 ] )';
            xcm = reshape( xcm_vec , [ 2 , obj.params.Nlinks ] )';
        end
        
        % theta2complex (converts an angle to a complex number)
        function complex = theta2complex( obj , theta )
            %theta2complex: Converts an angle relative to y-axis to a point on the complex unit circle
            %   Note that the answer is an array [a b] for the complex number a+ib
            
            a = sin( theta );
            b = cos( theta );
            
            complex = [ a , b ];
        end
        
        % complex_mult (multiply two complex numbers)
        function product = complex_mult( obj , z1 , z2 )
            %complex_mult: Multiply two complex numbers specified as vectors
            
            real = z1(1) * z1(1) - z1(2) * z2(1);
            im = z1(1) * z2(2) + z1(2) * z2(1);
            
            product = [ real , im ];
        end
        
        
        %% equations of motion
        
        % set_EOM
        function fcns = set_EOM(obj)
            %setEOM: Find symbolic expression of the equations of motion
            %   Also saves a function for evaluating the equations of motion
            
            %% define symbolic variables
            alpha = sym('alpha', [obj.params.Nlinks,1], 'real');    % joint angles
            alphadot = sym('alphadot', [obj.params.Nlinks,1], 'real');  % joint velocities
            alphaddot = sym('alphaddot', [obj.params.Nlinks,1], 'real');    % joint accel
            w = sym( 'w' , [2,1] , 'real' );    % load: [end effector mass ; gravity orientation]
            
            theta = obj.alpha2theta(alpha);
            thetadot = obj.alpha2theta(alphadot);
            
            [ x , xcm ]= obj.alpha2xvec(alpha);
            
            %% define Jacobians
            
            J_theta_alpha = jacobian( theta , alpha );
            
            J_xcm_alpha = jacobian( xcm , alpha );
            xcmdot = J_xcm_alpha * alphadot;    % velocity of link COMs
            
            J_x_alpha = jacobian( x , alpha );
            xdot = J_x_alpha * alphadot;    % velocity of ends of links
            xdot_eff = xdot(end-1:end); % velocity of end effector
            
            %% define useful matrices
            
            % mass matrix
            M = eye(obj.params.Nlinks) * obj.params.m;  % mass
            I = eye(obj.params.Nlinks) * obj.params.i;  % intertia
            K = ones(1 , obj.params.Nlinks) * obj.params.k; % stiffness
            D = eye(obj.params.Nlinks) * obj.params.d;  % damping
            
            %% define Lagrangian (L = KE - PE)
            
            % mass matrix
            m_joints = [ zeros( 2*(obj.params.Nlinks-1) , 1 ) ; w(1) ; w(1) ]; % point masses at the joints (end effector only)
            Dq = obj.params.m * J_xcm_alpha' * J_xcm_alpha...   % link masses
                 + obj.params.i * J_theta_alpha' * J_theta_alpha... % link inertias
                 + J_x_alpha(3:end,:)' * diag( m_joints ) * J_x_alpha(3:end,:);    % end effector mass
            
            % kinetic energy (links + end effector)
            KE = (1/2) * alphadot' * Dq * alphadot;
%             KE = (1/2) * alphadot' * Dq * alphadot + (1/2) * xdot_eff' * w(1) * xdot_eff;
            
            % potential energy (needs minus sign since "down" is positive)
%             PE = - obj.params.m * obj.params.g * ones(1 , length(xcm)/2) * xcm(2:2:end) + ...
%                 (1/2) * alpha' * obj.params.k * alpha;
%             h_links = sqrt( xcm(1:2:end).^2 + xcm(2:2:end).^2 ) .* sin( atan2( xcm(2:2:end) , xcm(1:2:end) ) - w(2) );
%             h_eff = sqrt( x(end-1)^2 + x(end)^2 ) * sin( atan2( x(end) , x(end-1) ) - w(2) );
%             h_links = ( tan(w(2)) .* xcm(1:2:end) - xcm(2:2:end) ) ./ sqrt( tan(w(2))^2 + 1 );
%             h_eff = ( tan(w(2)) .* x(end-1) - x(end) ) ./ sqrt( tan(w(2))^2 + 1 );
            xcm_mat = reshape( xcm , [ 2 , length(xcm)/2 ] )';
            h_links = xcm_mat * [ -sin( w(2) ) ; cos( w(2) ) ];
            h_eff = [ x(end-1) , x(end) ] * [ -sin( w(2) ) ; cos( w(2) ) ];
            PE = - obj.params.m * obj.params.g * ones(1 , length(xcm)/2) * h_links...   % links
                - w(1) * obj.params.g * h_eff...    % end effector
                + (1/2) * alpha' * obj.params.k * alpha;    % joint springs

            Lagrangian = KE - PE;
            
            %% derive equations of motion
            
            % save mass matrix as a function
            fcns.get_massMatrix = matlabFunction(Dq, 'Vars', { alpha , w }, 'Optimize', false);
            
            % derive non-inertial part of dynamics
            % creata a variable alpha that is a function of t
            syms t
            alpha_t = zeros( obj.params.Nlinks , 1 );
            alpha_t = sym(alpha_t);
            for i = 1 : obj.params.Nlinks
                istr = num2str(i);
                alpha_t(i) = str2sym(strcat('alpha_t', istr, '(t)'));
            end
            
            % write mass matrix as a function of t
            Dq_t = subs( Dq , alpha , alpha_t );
            
            % differentiate mass matrix wrt t
            Dq_dt = diff( Dq_t , t );
            
            % character substitutions to get rid of all the 'diff(x(t), t)' stuff
            alpha_dt = zeros( obj.params.Nlinks , 1 );
            alpha_dt = sym(alpha_dt);
            for i = 1 : obj.params.Nlinks
                istr = num2str(i);
                alpha_dt(i) = str2sym(strcat( 'diff(alpha_t', istr, '(t), t)' ));
            end
            Dq_dt = subs( Dq_dt , [ alpha_t , alpha_dt ] , [ alpha , alphadot ] ); % replace all t's
            
            dLdalpha = jacobian(Lagrangian, alpha)';
            
            % include damping and input terms
            % damping
            damp = obj.params.d * alphadot;
            fcns.get_damp = matlabFunction(damp, 'Vars', { alphadot }, 'Optimize', false);
            
            % input
            u = sym('u', [obj.params.Nmods,1], 'real'); % input. Desired joint angle for all joints in each module
            input = -obj.params.ku * ( kron( u , ones( obj.params.nlinks , 1) ) - alpha );   % vector of all joint torques
            fcns.get_input = matlabFunction(input, 'Vars', { alpha , u }, 'Optimize', false);
            
            % save damping and input as a function
            dampNinput = damp + input;
            fcns.get_dampNinput = matlabFunction(dampNinput, 'Vars', { alpha , alphadot , u }, 'Optimize', false);
            
            % save non-inertial part of dynamics as a function
            nonInert = Dq_dt * alphadot - dLdalpha + damp + input;
            fcns.get_nonInert = matlabFunction(nonInert, 'Vars', { alpha , alphadot , u , w });
        end
        
        % get_massMatrix
        function Dq = get_massMatrix( obj , alpha , w )
            % if no load is give, set it equal to zero
            if nargin < 2
                w = [ 0 ; 0 ];
            end
            Dq = obj.fcns.get_massMatrix( alpha , w );
        end
        
        % get_nonInert
        function nonInert = get_nonInert( obj , alpha , alphadot , u , w)
            % if no load is give, set it equal to zero
            if nargin < 4
                w = [ 0 ; 0 ];
            end
            nonInert = obj.fcns.get_nonInert( alpha , alphadot , u , w);
        end
        
        % get_damp
        function damp = get_damp( obj , alphadot )
            damp = obj.fcns.get_damp( alphadot );
        end
        
        % get_input
        function input = get_input( obj , alpha , u )
            input = obj.fcns.get_damp( alpha , u );
        end
        
        % get_dampNinput
        function dampNinput = get_dampNinput( obj , alpha , alphadot , u )
            dampNinput = obj.fcns.get_dampNinput( alpha , alphadot , u );
        end
        
        % vf_RHS (dynamics as vector field)
        function RHS = vf_RHS( obj , t , Alpha , u , w)
            %vf_RHS: RHS of EoM, Dq(x) * xddot = C(x,xdot) * xdot + g(x)
            % with appropriate dimension to work in state space formulation
            %   Alpha = [ alpha ; alphadot ];
            %   Alphadot = [ alphadot ; alphaddot ];
            %   note: to use with ode45, need to include the mass matrix in
            %         the call to ode45
            
            % if no load is give, set it equal to zero
            if nargin < 4
                w = [ 0 ; 0 ];
            end
            
            params = obj.params;
            
            alpha = Alpha( 1 : params.Nlinks );
            alphadot = Alpha( params.Nlinks+1 : end );
            
            nonInert = obj.get_nonInert( alpha , alphadot , u , w);
            
            RHS = -[ -alphadot ; nonInert ];
        end
        
        % vf_massMatrix (dynamics as vector field)
        function D = vf_massMatrix( obj , t , Alpha , u , w )
            %vf_massMatrix: mass matrix with appropriate dimension to work 
            % in state space formulation 
            %   Alpha = [ alpha ; alphadot ];
            %   Alphadot = [ alphadot ; alphaddot ];
            %   note: to use with ode45, need to include the mass matrix in
            %         the call to ode45
            
            % if no load is give, set it equal to zero
            if nargin < 4
                w = [ 0 ; 0 ];
            end
            
            params = obj.params;
            
            alpha = Alpha( 1 : params.Nlinks );
            alphadot = Alpha( params.Nlinks+1 : end );
            
            Dq = obj.get_massMatrix( alpha , w);
            
            D = blkdiag( eye(params.Nlinks) , Dq );
        end
        
        %% sensing
        
        % get_markers (simulated mocap)
        function markers = get_markers( obj , alpha )
            [ x , ~ ] = obj.alpha2x( alpha );
            markers = x( 1 : obj.params.nlinks : end , : );
        end
        
        % points2poly (convert marker points into a polynomial)
        function [coeffs, obs_matrix] = points2poly(obj, degree, points, positions, orient)
            %points2poly: Finds polynomial that best goes through a set of points.
            % Polynomial starts at the origin, and its domain is [0,1].
            % "Resting configuration" is along the yaxis (2d) or zaxis (3d)
            %   degree - scalar, maximum degree of the polynomial
            %   points - matrix, rows are xy(z) coordinates of points
            %   positions - (row) vector of [0,1] positions of points on the arm.
            %   orient - (row) vector, orientation of the the end effector (complex number for 2d case, quaternion for 3d case)
            %   coeffs - matrix, rows are the coefficients of the polynomial in each
            %            coordinate where [a b c ...] -> ax^1 + bx^2 + cx^2 + ...
            %   obs_matrix - matrix that converts from state vector to coefficients
            
            % for the 2d case (will consider the 3d case later)
            if size(points,2) == 2
                if all( size(orient) ~= [1,2] )
                    error('orientation for 2d system must be a complex number specified as [1x2] vector');
                end
                
                % generate virtual points to provide slope constraint at the base & end
                startpoint = [ 0 , 1e-2 ];
%                 endpoint = obj.complex_mult( orient/norm(orient) , [ 0 , 1 ] )*1e-2 + points(end,:);  
                endpoint = ( orient/norm(orient) )*1e-2 + points(end,:);    % add point extended from last link
                points_supp = [0 , 0 ; startpoint ; points ; endpoint];
                %     points_supp = points;   % remove the slope constraints
                
                % generate A matrix for least squares problem
                positions_supp = [ 0 , 1e-2 , positions , 1+1e-2 ];
                %     positions_supp = positions;   % remove the slope constraints
                A = zeros( length(positions_supp) , degree );
                for i = 1 : degree
                    A(:,i) = reshape( positions_supp , [] , 1) .^ i;
                end
                
                % separate x and y corrdinates of points
                points_x = points_supp(:,1);
                points_y = points_supp(:,2);
                
                % find polynomial coefficients
                obs_matrix = pinv(A);
                coeffs_vec_x = obs_matrix * points_x;
                coeffs_vec_y = obs_matrix * points_y;
                
                % make coeffs a matrix to where each row is coeffs for one dimension
                coeffs = [ coeffs_vec_x' ; coeffs_vec_y' ];
            else
                error('points matrix must be nx2');
            end
        end
        
        % get_y (extracts the measured output from the full state)
        function y = get_y( obj , x )
            %get_y: Gets the output from the state (in this case Alpha)
            %   x - array containing one or more state values. Each row
            %    should be a state.
            %   y - array containing the corresponding output values. Each
            %    row is an output.
            
            % check that input is the correct size
            if size( x , 2 ) ~= obj.params.Nlinks * 2
                error(['Input state matrix has wrong dimension. ' ,...
                    'Its width should be ' , num2str(obj.params.Nlinks*2) ,...
                    ' You may need to take its transpose.']);
            end
            
            % set the output, depending on what the output_type is.
            if strcmp( obj.output_type , 'markers' )
%                 y = zeros( size(x,1) , 2 * ( obj.params.Nlinks ) + 2 );
                y = zeros( size(x,1) , 2 * ( obj.params.Nlinks ) );
                for i = 1 : size(x,1)
                    alpha = x( i , 1 : obj.params.Nlinks );
%                     theta = obj.alpha2theta( alpha );
                    temp = obj.get_markers( alpha );
                    markers = reshape( temp' , [ 1 , 2 * ( obj.params.Nmods+1 ) ] );
%                     orient = obj.theta2complex( theta(end) );
%                     y(i,:) = [ markers( : , 3:end ) , orient ]; % (remove 0th marker position because it is always at the origin)
                    y(i,:) = markers( : , 3:end ); % (remove 0th marker position because it is always at the origin)
                end
            elseif strcmp( obj.output_type , 'angles' )
                y = zeros( size(x,1) , obj.params.Nlinks );
                for i = 1 : size(x,1)
                    y(i,:) = x( i , 1 : obj.params.Nlinks );    % first half of x is the joint angles
                end
            elseif strcmp( obj.output_type , 'endeff' )
                y = zeros( size(x,1) , 2 ); % assumes 2D arm
                for i = 1 : size(x,1)
                    alpha = x( i , 1 : obj.params.Nlinks );
                    temp = obj.get_markers( alpha );
                    markers = reshape( temp' , [ 1 , 2 * ( obj.params.Nmods+1 ) ] );
                    y(i,:) = markers( : , end-1:end ); % (only keep the last marker position)
                end
            elseif strcmp( obj.output_type , 'shape' )
                y = zeros( size(x,1) , 2*3 ); % assumes planar arm, 3rd order polynomial
                for i = 1 : size(x,1)
                    alpha = x( i , 1 : obj.params.Nlinks );
                    shapecoeffs = obj.get_shape_coeffs( alpha , 3 );
                    y(i,:) = shapecoeffs;
                end
            end
        end
        
        % get_shape
        function [ shape , coeffs ] = get_shape( obj , alpha , degree)
            points = get_markers( obj , alpha );   % coordinates of mocap markers
            positions = obj.params.markerPos;    % relative location of markers on arm [0,1]
            theta = obj.alpha2theta( alpha );
            orient = obj.theta2complex( theta(end) );    % orientaton of end effector
            coeffs = obj.points2poly( degree , points(2:end,:) , positions(2:end) , orient );    % convert points of a polynomial (skip the origin)
            
            % get the shape
            px = fliplr( [ 0 , coeffs(1,:) ] );
            py = fliplr( [ 0 , coeffs(2,:) ] );
            
            pol_x = zeros(101,1); pol_y = zeros(101,1);
            for i = 1:101
                pol_x(i) = polyval(px,0.01*(i-1));
                pol_y(i) = polyval(py,0.01*(i-1));
            end
            shape = [ pol_x , pol_y ];
        end
        
        % get_shape_coeffs (gets the shape coefficients for many points)
        function coeffs = get_shape_coeffs( obj , alpha , degree )
            %get_shape_coeffs: gets the shape coefficients for many points
            %   alpha: rows should be individual configurations
            %   degree: degree of the shape polynomial to be fit
            %   coeffs: returns the shape coefficients row vectors
            
            coeffs = zeros( size(alpha,1) , degree * 2 );   % assumes 2d-planar arm
            for i = 1 : size( alpha , 1 )
                [ ~ , coeff_matrix ] = obj.get_shape( alpha(i,:) , degree );
                coeff_vec = reshape( coeff_matrix' , [ numel( coeff_matrix ) , 1 ] )';
                coeffs(i,:) = coeff_vec;
            end
        end
        
        
        %% visualization
        
        % def_fig (defines a default figure for plotting arm
        function fig = def_fig( obj )
            %def_fig: set up a figure for plotting
            fig = figure;
            axis([-obj.params.L, obj.params.L, -0.5*obj.params.L, 1.5*obj.params.L])
            set(gca,'Ydir','reverse')
            xlabel('x(m)')
            ylabel('y(m)')
        end
        
        % plot_arm
        function ph = plot_arm( obj , alpha )
            % convert to xy-coordinates
            [ X , ~ ] = obj.alpha2x( alpha );
            x = X(:,1);
            y = X(:,2);
            
            % add markers
            markers = obj.get_markers( alpha );
            
            % plot it
            hold on
            ph(1) = plot( x, y, '-o' );
            ph(2) = plot( markers(:,1) , markers(:,2) , 'r*');
            hold off
        end
        
        % plot_arm_shape
        function ph = plot_arm_shape( obj , alpha , degree )
            % plot the arm
            ph = obj.plot_arm( alpha );
            
            % get the shape
            [ shape , ~ ] = obj.get_shape( alpha , degree );
            
            % plot it
            hold on
            ph(3) = plot( shape(:,1) , shape(:,2) , 'r');
            hold off
        end
        
        % animate_arm
        function animate_arm( obj, t , y , w , name )
            %animate_arm: Animate a simualtion of the arm
            %   t - time vector from simulation
            %   y - state vector from simulation (alpha and alphadot)
            %   varargin{1} = degree - degree of the shape polynomial (default: 3)
            %   varargin{2} = name - name of the video file (default: sysName)

            % deal with optional arguments w and name
            if isempty(w)
                w = kron( ones( size(t) ) , [0,0] );   % default is no load
            elseif size( w , 1 ) == 1
                w = kron( ones( size(t) ) , w );   % constant load   
            end
            if ~exist( 'name' , 'var')
                name = obj.params.sysName;
            end
            
            alpha = y(: , 1:obj.params.Nlinks );   % joint angles over time
            
%             fig = figure;   % create figure for the animation
            fig = figure('units','pixels','position',[0 0 720 720]);   % create figure for the animation (ensures high resolution)
            axis([-1.1*obj.params.L, 1.1*obj.params.L, -1.1*obj.params.L, 1.1*obj.params.L])
            set(gca,'Ydir','reverse')
            xlabel('$\hat{\alpha}$ (m)' , 'Interpreter' , 'Latex' , 'FontSize' , 26);
            ylabel('$\hat{\beta}$ (m)' , 'Interpreter' , 'Latex' , 'FontSize' , 26);
            daspect([1 1 1]);   % make axis ratio 1:1
           
            % Prepare the new file.
            vidObj = VideoWriter( ['animations' , filesep , name , '.mp4'] , 'MPEG-4' );
            vidObj.FrameRate = 30;
            open(vidObj);
            
            set(gca,'nextplot','replacechildren', 'FontUnits' , 'normalized');
            
            totTime = t(end);    % total time for animation (s)
            nsteps = length(t); % total steps in the simulation
            totFrames = 30 * totTime;   % total frames in 30 fps video
            
            % Grid points for gravity direction arrows
            arrow_len = 0.1;
            [ x_grid , y_grid ] = meshgrid( -1.25*obj.params.L:arrow_len:1.25*obj.params.L , -1.25*obj.params.L:arrow_len:1.25*obj.params.L);
            
            % run animation fram by frame
            for i = 1:totFrames
                
                index = floor( (i-1) * (nsteps / totFrames) ) + 1;   % skips points between frames
                
                % direction of gravity
                u_grid = -ones( size(x_grid) ) * arrow_len * sin(w(index,2));
                v_grid = ones( size(y_grid) ) * arrow_len * cos(w(index,2));
                
                % locations of joints
                [ X , ~ ] = obj.alpha2x( alpha(index,:)' );
                x = X(:,1);
                y = X(:,2);
                marker = obj.get_markers( alpha(index,:) );   % get mocap sensor location
                
                hold on;
                p0 = quiver( x_grid , y_grid , u_grid , v_grid , 'Color' , [0.75 0.75 0.75] );
                p1 = plot(x, y, 'k-o' , 'LineWidth' , 3);
%                 p2 = plot( marker(:,1) , marker(:,2) , 'r*');
                p4 = plot( marker(end,1) , marker(end,2) , 'bo' , 'MarkerSize' , 2*w(index,1) + 0.01 , 'MarkerFaceColor' , 'b' );  % end effector load
                hold off;
                grid on;
                
                % write each frame to the file
                currFrame = getframe(fig);
                writeVideo(vidObj,currFrame);
                
                delete(p0);
                delete(p1);
%                 delete(p2); 
                delete(p4);
            end
            
            close(vidObj);
        end
        
        % animate_arm_refvmpc
        function animate_arm_refvmpc( obj, t , xref , xmpc , varargin)
            %animate_arm: Animate a simualtion of the arm
            %   t - time vector from simulation
            %   xref - state reference vector (alpha and alphadot)
            %   xmpc - state vector from simulation (alpha and alphadot)
            %   varargin{1} = degree - degree of the shape polynomial (default: 3)
            %   varargin{2} = name - name of the video file (default: sysName)
            
            % deal with optional inputs
            if length(varargin) == 2
                degree = varargin{1};
                name = varargin{2};
            elseif length(varargin) == 1
                degree = varargin{1};
                name = obj.params.sysName;
            else
                degree = 3;
                name = obj.params.sysName;
            end
            
            alpha_ref = xref(: , 1:obj.params.Nlinks );   % joint angles over time
            alpha_mpc = xmpc(: , 1:obj.params.Nlinks );   % joint angles over time
            
            fig = figure;   % create figure for the animation
            axis([-1.25*obj.params.L, 1.25*obj.params.L, -1.25*obj.params.L, 1.25*obj.params.L])
            set(gca,'Ydir','reverse')
            xlabel('x(m)')
            ylabel('y(m)')
            
            % Prepare the new file.
            vidObj = VideoWriter( ['animations' , filesep , name , '.mp4'] , 'MPEG-4' );
            vidObj.FrameRate = 30;
            open(vidObj);
            
            set(gca,'nextplot','replacechildren', 'FontUnits' , 'normalized');
            
            totTime = t(end);    % total time for animation (s)
            nsteps = length(t); % total steps in the simulation
            totFrames = 30 * totTime;   % total frames in 30 fps video
            
            % run animation frame by frame
            for i = 1:totFrames
                
                index = floor( (i-1) * (nsteps / totFrames) ) + 1;   % skips points between frames
                
                % plot the reference arm
                [ Xref , ~ ] = obj.alpha2x( alpha_ref(index,:)' );
                x_ref = Xref(:,1);
                y_ref = Xref(:,2);
                marker_ref = obj.get_markers( alpha_ref(index,:) );   % get mocap sensor location
                [shape_ref , ~ ] = obj.get_shape( alpha_ref(index,:) , degree); % get polynomial approx of shape
                
                hold on;
                p1 = plot(x_ref, y_ref, '-o' , 'Color' , [0 0 1 0.5] );
                p2 = plot( marker_ref(:,1) , marker_ref(:,2) , '*' , 'Color' , [1 0 0 0.5]);
                p3 = plot( shape_ref(:,1) , shape_ref(:,2) , 'Color' , [1 0 0 0.5]);
                hold off;
                
                % plot the actual arm
                [ Xmpc , ~ ] = obj.alpha2x( alpha_mpc(index,:)' );
                x_mpc = Xmpc(:,1);
                y_mpc = Xmpc(:,2);
                marker_mpc = obj.get_markers( alpha_mpc(index,:) );   % get mocap sensor location
                [shape_mpc , ~ ] = obj.get_shape( alpha_mpc(index,:) , degree); % get polynomial approx of shape
                
                hold on;
                p4 = plot(x_mpc, y_mpc, '-o' , 'Color' , [0 0 1 1] );
                p5 = plot( marker_mpc(:,1) , marker_mpc(:,2) , '*' , 'Color' , [1 0 0 1]);
                p6 = plot( shape_mpc(:,1) , shape_mpc(:,2) , 'Color' , [1 0 0 1]);
                hold off;
                
                % write each frame to the file
                currFrame = getframe(fig);
                writeVideo(vidObj,currFrame);
                
                delete(p1); delete(p2); delete(p3);
                delete(p4); delete(p5); delete(p6);
            end
            
            close(vidObj);
        end
        
        % animate_arm_refendeff
        function animate_arm_refendeff( obj, t , ref , y , w , name )
            %animate_arm_refendeff: Animate a simualtion of the arm
            % and show the desired end effector trajectory
            %   t - time vector from simulation
            %   y - state vector from simulation (alpha and alphadot)
            %   varargin{1} = degree - degree of the shape polynomial (default: 3)
            %   varargin{2} = name - name of the video file (default: sysName)

            % deal with optional arguments w and name
            if isempty(w)
                w = kron( ones( size(t) ) , [0,0] );   % default is no load
            elseif size( w , 1 ) == 1
                w = kron( ones( size(t) ) , w );   % constant load   
            end
            if ~exist( 'name' , 'var')
                name = obj.params.sysName;
            end
            
            % colormap
            colormap lines;
            cmap = colormap;
            linewidth = 5;  % width of the lines
            pathwidth = 2;  % width of end effector path line
            
            alpha = y(: , 1:obj.params.Nlinks );   % joint angles over time
            
%             fig = figure;   % create figure for the animation
            fig = figure('units','pixels','position',[0 0 720 720]);   % create figure for the animation (ensures high resolution)
%             axis([-1.25*obj.params.L, 1.25*obj.params.L, -1.25*obj.params.L, 1.25*obj.params.L]);
            window_buffer = 0.5;
            window = [ min(ref(:,1))-window_buffer , max(ref(:,1))+window_buffer , min(ref(:,2))-window_buffer , max(ref(:,2))+window_buffer-0.3 ]; % axis limits
            axis(window);
            set(gca,'Ydir','reverse')
            xlabel('$\hat{\alpha}$ (m)' , 'Interpreter' , 'Latex' , 'FontSize' , 26);
            ylabel('$\hat{\beta}$ (m)' , 'Interpreter' , 'Latex' , 'FontSize' , 26);
            daspect([1 1 1]);   % make axis ratio 1:1
           
            % Prepare the new file.
            vidObj = VideoWriter( ['animations' , filesep , name , '.mp4'] , 'MPEG-4' );
            vidObj.FrameRate = 30;
            open(vidObj);
            
            set(gca,'nextplot','replacechildren', 'FontUnits' , 'normalized');
            
            totTime = t(end);    % total time for animation (s)
            nsteps = length(t); % total steps in the simulation
            totFrames = 30 * totTime;   % total frames in 30 fps video
            
            % Grid points for gravity direction arrows
            arrow_len = 0.1;
%             [ x_grid , y_grid ] = meshgrid( -1.25*obj.params.L:arrow_len:1.25*obj.params.L , -1.25*obj.params.L:arrow_len:1.25*obj.params.L);
            [ x_grid , y_grid ] = meshgrid( window(1):arrow_len:window(2) , window(3):arrow_len:window(4));
            
            % initialize end effector path
            endeff_path = zeros(totFrames,2);
            
            % run animation fram by frame
            for i = 1:totFrames
                
                index = floor( (i-1) * (nsteps / totFrames) ) + 1;   % skips points between frames
                
                % plot the reference trajectory
                x_ref = ref(:,1);
                y_ref = ref(:,2);
                hold on;
                r1 = plot(x_ref , y_ref, '-' , 'Color' , [0 0 0 0.5] , 'Linewidth' , linewidth );
                r2 = plot( x_ref(index) , y_ref(index) , '*' , 'Color' , [1 0 0]);
                hold off;
                
                % direction of gravity
                u_grid = -ones( size(x_grid) ) * arrow_len * sin(w(index,2));
                v_grid = ones( size(y_grid) ) * arrow_len * cos(w(index,2));
                
                % locations of joints
                [ X , ~ ] = obj.alpha2x( alpha(index,:)' );
                x = X(:,1);
                y = X(:,2);
                marker = obj.get_markers( alpha(index,:) );   % get mocap sensor location
                
                % remember path of the end effector
                endeff_path(i,:) = marker(end,:);
                
                hold on;
                p0 = quiver( x_grid , y_grid , u_grid , v_grid , 'Color' , [0.75 0.75 0.75] );
                p1 = plot(x, y, 'k-o' , 'LineWidth' , linewidth);
%                 p2 = plot( marker(:,1) , marker(:,2) , 'r*');
                p4 = plot( marker(end,1) , marker(end,2) , 'bo' , 'MarkerSize' , 20*w(index,1) + 0.01 , 'MarkerFaceColor' , 'b' );  % end effector load
                p5 = plot( endeff_path(1:i,1) , endeff_path(1:i,2) , 'Color' , cmap(4,:) , 'LineWidth' , pathwidth);     % end effector path
                hold off;
                grid on;
                
                % write each frame to the file
                currFrame = getframe(fig);
                writeVideo(vidObj,currFrame);
                
                delete(r1);
                delete(r2);
                delete(p0);
                delete(p1);
%                 delete(p2); 
                delete(p4);
                delete(p5);
            end
            
            close(vidObj);
        end
        
            
        %% simulation
        
        % simulate system under random "ramp and hold" inputs
        function sim = simulate_rampNhold( obj , tf , Tramp , w , varargin)
            %simulate_rampNhold: simulate system under random "ramp and hold" inputs
            %   tf - length of simulation(s)
            %   Tramp - ramp period length
            %   w - load condition [ endeff mass , angle of gravity (normal is 0) ]
            %       (assumed constant for entire trial)
            %   Alpha - joint angles and velocities at each timestep
            %   markers - marker position at each time step [x0,y0,...,xn,yn ; ...]
            %   varargin - save on? (true/false)
            
%             obj.params.umax = 4.5*pi/8;   % DEBUG, make inputs larger
            
            % replace default values with user input values
            p = inputParser;
            addParameter( p , 'saveon' , false );
            parse( p , varargin{:} );
            saveon = p.Results.saveon;
            
            % time steps
            tsteps = ( 0 : obj.params.Ts : tf )';    % all timesteps
            tswitch = ( 0 : Tramp : tf )';  % input switching points
            
            % table of inputs
            numPeriods = ceil( length(tswitch) / 2 );
            inputs_nohold = obj.params.umax .* ( 2*rand( numPeriods , obj.params.Nmods ) - 1 );  % table of random inputs
            inputs_hold = reshape([inputs_nohold(:) inputs_nohold(:)]',2*size(inputs_nohold,1), []); % repeats rows of inputs so that we get a hold between ramps
            u = interp1( tswitch , inputs_hold( 1:length(tswitch) , : ) , tsteps , 'linear' , 0 );
            
            % initial condition (resting)
            a0 = zeros( obj.params.Nlinks , 1 );
            adot0 = zeros( obj.params.Nlinks , 1 );
            
            % simulate system
            options = odeset( 'Mass' , @(t,x) obj.vf_massMatrix( t , x , u , w' ) );
            [ t , Alpha ] = ode45( @(t,x) obj.vf_RHS( t , x , u( floor(t/obj.params.Ts) + 1 , : )' , w' ) , tsteps , [ a0 ; adot0 ] , options );    % with mass matrix, variable time step
            
            % get locations of the markers at each time step
            markers = zeros( length(t) , 2 * ( obj.params.Nmods+1 ) );
            orient = zeros( length(t) , 2 );
            for i = 1 : size(Alpha,1)
                alpha = Alpha( i , 1 : obj.params.Nlinks );
                theta = obj.alpha2theta( alpha );
                temp = obj.get_markers( alpha );
                markers(i,:) = reshape( temp' , [ 1 , 2 * ( obj.params.Nmods+1 ) ] );
                orient(i,:) = obj.theta2complex( theta(end) ); 
            end
            
            % define output
            sim.t = t;  % time vector
            sim.x = Alpha;  % internal state of the system
            sim.alpha = Alpha( : , 1 : obj.params.Nlinks );   % joint angles
            sim.alphadot = Alpha( : , obj.params.Nlinks+1 : end );  % joint velocities
            sim.y = obj.get_y( Alpha );    % output based on available observations (remove 0th marker position because it is always at the origin)
            sim.u = u;  % input
            sim.w = kron( ones(size(t)) , w );  % load condition
            sim.params = obj.params;    % parameters associated with the system
            
            % save results
            if saveon
                fname = [ 'systems' , filesep , obj.params.sysName , filesep , 'simulations' , filesep , 'tf-', num2str(tf) , 's_ramp-' , num2str(Tramp) , 's.mat' ];
                unique_fname = auto_rename( fname , '(0)' );
                save( unique_fname , '-struct' ,'sim' );
            end
        end
           
        % simulate_Ts (simulate system over a single time step)
        function [ x_kp1 ] = simulate_Ts( obj , x_k , u_k , w_k , tstep )
            %simulate_Ts: Simulate system over a single time step
            %   x_k - current value of state (full state Alpha = [alpha ; alphadot])
            %   u_k - input over next time step
            %   w_k - load condition over next time step
            
            % if load field is empty, set it to zero
            if isempty(w_k)
                w_k = [ 0 ; 0 ];
            end
            
            % if no timestep is provided use the default stored in params
            if nargin < 5
                tspan = [ 0 , obj.params.Ts ];
            else
                tspan = [ 0 , tstep ];
            end
            
            % simulate system
            options = odeset( 'Mass' , @(t,x) obj.vf_massMatrix( t , x , u_k , w_k ) );
            [ t , Alpha ] = ode45( @(t,x) obj.vf_RHS( t , x , u_k , w_k ) , tspan , x_k , options );    % with mass matrix, variable time step
            
            % set output, the state after one time step
            x_kp1 = Alpha(end,:)';
        end
        
        % simulate (simulate system given time and input vector)
        function sim = simulate( obj , t_in , u_in , w_in , varargin)
            %simulate_zoh: simulate system given a vector of timesteps, 
            % inputs, and loads
            %   t_in - time vector
            %   u_in - input vector (rows are distinct inputs)
            %   w_in - load vector (must have 2 columns)
            %   varargin - 'Name' , 'Value' pairs
            %       'saveon' - (true/false)
            %       'input_type' - 'zoh' , 'interp' 
                      
            % replace default values with user input values
            p = inputParser;
            addParameter( p , 'saveon' , false );
            addParameter( p , 'input_type' , 'zoh' );
            parse( p , varargin{:} );
            saveon = p.Results.saveon;
            input_type = p.Results.input_type;
            
            % set value of the load if none provided
            if ~exist('w_in','var')
                w_in = zeros( length(t_in) , 2 );
            elseif size(w_in,1) == 1   % if w is only given for one timestep
                w_in = kron( ones( size(t_in) ) , w_in ); % stack it repeadedly
            elseif size(w_in,2) ~= 2
                error('w_in must have width of 2');
            end
            
            % verify that the dimension of the inputs are correct
            numsteps = length(t_in);
            if size(u_in,1) ~= numsteps
                error('t_in and u_in vectors need to be the same length');
            elseif size(u_in,2) ~= obj.params.Nmods
                error('u_in width must be the same as the number of modules');
            elseif size(t_in,2) ~= 1
                error('t_in must be a column vector');
            end
           
            
            % initial condition (resting, i.e. all joint angles are zero)
            a0 = zeros( obj.params.Nlinks , 1 );
            adot0 = zeros( obj.params.Nlinks , 1 );
            
            % simulate system
            options = odeset( 'Mass' , @(t,x) obj.vf_massMatrix( t , x , [] , w_in( obj.get_k(t,t_in) + 1 , : )' ) );   % u has no effect on mass mtx so argument is []
            if strcmp( input_type , 'zoh' )
                [ t_out , Alpha ] = ode45( @(t,x) obj.vf_RHS( t , x , u_in( obj.get_k(t,t_in) + 1 , : )' , w_in( obj.get_k(t,t_in) + 1 , : )' ) , t_in , [ a0 ; adot0 ] , options );    % with mass matrix, variable time step
            elseif strcmp( input_type , 'interp' )
                [ t_out , Alpha ] = ode45( @(t,x) obj.vf_RHS( t , x , ...
                    ( u_in(obj.get_k(t,t_in)+1,:) + ( u_in(obj.get_k(t,t_in)+2,:) - u_in(obj.get_k(t,t_in)+1,:) ) / ( t_in(obj.get_k(t,t_in)+2) - t_in(obj.get_k(t,t_in)+1) ) * ( t - t_in(obj.get_k(t,t_in)+1)) )' , ... % interpolated input
                    w_in( obj.get_k(t,t_in) + 1 , : )' ) ,... % load (not-interpolated)
                    t_in(1:end-1) , [ a0 ; adot0 ] , options );    % with mass matrix, variable time step
            else
                error('input_type argument is not valid. Choices are <zoh> or <interp>.');
            end
                
            % get locations of the markers at each time step
            markers = zeros( length(t_out) , 2 * ( obj.params.Nmods+1 ) );
            orient = zeros( length(t_out) , 2 );
            for i = 1 : size(Alpha,1)
                alpha = Alpha( i , 1 : obj.params.Nlinks );
                theta = obj.alpha2theta( alpha );
                temp = obj.get_markers( alpha );
                markers(i,:) = reshape( temp' , [ 1 , 2 * ( obj.params.Nmods+1 ) ] );
                orient(i,:) = obj.theta2complex( theta(end) ); 
            end
            
            % define output
            sim.t = t_out;  % time vector
            sim.x = Alpha;  % internal state of the system
            sim.alpha = Alpha( : , 1 : obj.params.Nlinks );   % joint angles
            sim.alphadot = Alpha( : , obj.params.Nlinks+1 : end );  % joint velocities
            sim.y = obj.get_y( Alpha );    % output based on available observations (remove 0th marker position because it is always at the origin)
            sim.u = u_in( 1 : length(t_out) , : );  % input
            sim.w = w_in;
            sim.params = obj.params;    % parameters associated with the system
            
            % save results
            if saveon
                fname = [ 'systems' , filesep , obj.params.sysName , filesep , 'simulations' , filesep , 'tf-', num2str(tf) , 's_ramp-' , num2str(Tramp) , 's.mat' ];
                unique_fname = auto_rename( fname , '(0)' );
                save( unique_fname , '-struct' ,'sim' );
            end
        end  
            
        % get_k (find the time array index of a specific time)
        function k = get_k( obj , t , t_vec )
            if t == 0
                index = find( t_vec <= t ); 
            else
                index = find( t_vec < t );
            end
            k = index(end);
        end
        
        % get_rampNhold (give rampNhold signal between an upper and lower bound)
        function [ signal , tsteps ] = get_rampNhold( obj , tf , Tramp , lb , ub)
            %get_rampNhold: simulate system under random "ramp and hold" inputs
            %   tf - length of simulation(s)
            %   Tramp - ramp period length
            %   lb - [1 x dim of signal], lower bound for the signal
            %   ub - [1 x dim of signal], upper bound for the signal
            
            % time steps
            tsteps = ( 0 : obj.params.Ts : tf )';    % all timesteps
            tswitch = ( 0 : Tramp : tf )';  % input switching points
            
            % table of inputs
            numPeriods = ceil( length(tswitch) / 2 );
            signal_nohold = ( ub - lb ) .* rand( numPeriods , length(lb) ) + lb ;  % table of random inputs
            signal_hold = reshape([signal_nohold(:) signal_nohold(:)]',2*size(signal_nohold,1), []); % repeats rows of inputs so that we get a hold between ramps
            signal = interp1( tswitch , signal_hold( 1:length(tswitch) , : ) , tsteps , 'linear' , 0 );
        end
        
        
    end  
end











