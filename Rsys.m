classdef Rsys
    %Rsys: Class that generates and simulates sytems with randomly
    %generated nonlinear dynamics
    %   Generated systems have a 1-dimensional state and input (FOR NOW)
    
    properties
        num_sys;    % number of systems to generate
        degree_x;   % state monomial degree in dynamics of system (could be 2*degree)
        degree_u;   % input monomial degree in dynamics of system (could be 2*degree)
        num_terms;  % number of terms in dynamics of system
        
        systems;    % cell array containing symbolic expressions and functions of dynamics for all systems
    end
    
    methods
        function obj = Rsys( num_sys_in , num_terms_in , degree_x_in , degree_u_in )
            %CLASS CONSTRUCTOR: Construct an instance of this class
            %   Detailed explanation goes here
            
            % set property values from inputs to constructor
            obj.num_sys = num_sys_in;
            obj.num_terms = num_terms_in;
            obj.degree_x = degree_x_in;
            obj.degree_u = degree_u_in;
            
            % initialize other parameters
            obj.systems = cell( obj.num_sys , 1 );
            obj = obj.construct_systems;
        end
        
        %% Constructing systems
        
        % construct_systems
        function obj = construct_systems( obj )
            %construct_systems: Self explanatory funcion name
            %   Detailed explanation goes here
            
            % define monomials of specified degree
            syms t x u real
            x_monomial = sym( ones( 1 , obj.degree_x + 1 ) );
            u_monomial = sym( ones( 1 , obj.degree_u + 1 ) );
            for i = 1 : obj.degree_x + 1
                x_monomial(1,i) = x_monomial(1,i) * x^(i-1);
            end
            for i = 1 : obj.degree_u + 1
                u_monomial(1,i) = u_monomial(1,i) * u^(i-1);
            end
            
            % define dictionary of functions 
            funcs = [ x_monomial , sin(x) , cos(x) , u_monomial , sin(u) , cos(u) ];
            
            for i = 1 : obj.num_sys
                % define random coefficients and combinations
                coeffs = 1*( 2*rand(obj.num_terms,1) - 1 );
                selectors = randi( [0,1] , obj.num_terms , obj.degree_x + obj.degree_u + 2 + 4 );
%                 selectors = ones( obj.num_terms , obj.degree_x + obj.degree_u + 2 + 4 );    % use all terms
                
                % define terms
                terms = sym( ones(1,obj.num_terms) );
                for j = 1 : obj.num_terms
                    terms(j) = coeffs(j) * prod( funcs.^selectors(j,:) );
                end
                
                % dynamics are defined as sum of the terms times gaussian
                xdot = sum( terms ) * exp(-x^2);
                
                % define output
                obj.systems{i}.vf_sym = xdot;
                obj.systems{i}.vf_func = matlabFunction( xdot , 'Vars' , {t,x,u} );
            end
        end
        
        %% simulating systems
        
        % simulate_systems
        function data = simulate_systems( obj , t_end , Ts , num_trials , x0 )
           % simulate_systems
           %    t_end - length of trials (seconds)
           %    Ts - sampling time (seconds)
           %    num_trials - number of trials to simulate for each system
           %    x0 - single or set of initial conditions ( num_trials , nx )
           
           % make sure initial condition is correct dimension
           if size( x0 , 1 ) == 1
               x0 = kron( ones( num_trials , 1 ) , x0 );
           end
           
           % define sampling points
           tq = ( 0 : Ts : t_end )';
           
           data = cell( num_trials , obj.num_sys );
           for i = 1 : obj.num_sys
               for j = 1 : num_trials
                   % generate random inputs between [-1,1]
                   uq = 2 * rand( length(tq) , 1 ) - 1;
%                    uq = obj.generate_input_steps( tq , 100 );
                   
                   % DO ODE45 IN HERE
                   [ t_out , y_out ] = ode45( @(t,x) obj.systems{i}.vf_func( t , x , obj.get_u(t,tq,uq) ) , tq , x0(j,:)' );
                   data{j,i}.t = t_out;
                   data{j,i}.y = y_out;
                   data{j,i}.u = uq;
               end
           end
        end
        
        % get_u
        function u = get_u( obj , t , tq , uq )
           % get_u: Select current input from sequence based on time index
           steps_elapsed = find( tq <= t );
           index = steps_elapsed(end);
           u = uq(index,:);
        end
        
        % generate_input_ramps
        function U = generate_input_steps( obj , tq , num_steps_ramp )
            % generate_input_ramps
            %   Generate a sequence of inputs made up of steps, 
            %   rather than just comeletely random inputs. Should allow for
            %   more system exitation/movement I think
            
            ind = 1 : num_steps_ramp : length( tq );
            inputs = 2 * rand( length(ind) , 1 ) - 1;
            
            U = zeros( size(tq) );  % only works for 1-D input
            for i = 1 : length( ind ) - 1
                U( ind(i) : ind(i+1)-1 ) = inputs( i );
            end
        end
        
        % plot_data
        function plot_data( obj , data )
            % plot_data: Plot data, one plot for each system
            for i = 1 : size( data , 2 )
                figure;
                title( [ 'System ' , num2str(i) ] );
                
                % Output plot
                subplot(2,1,1);
                hold on;
                for j = 1 : size( data , 1 )
                    plot( data{j,i}.t , data{j,i}.y );
                end
                hold off;
                xlabel('t');
                ylabel('y');
                
                % Input plot
                subplot(2,1,2);
                hold on;
                for j = 1 : size( data , 1 )
                    plot( data{j,i}.t , data{j,i}.u );
                end
                hold off;
                xlabel('t');
                ylabel('u')
            end
        end
        
        % save_data
        function save_data( obj , data )
            % save_data: save all of the data in a format that is usable
            % with the Ksysid class
            
            % create name of folder
            dateString = datestr(now , 'yyyy-mm-dd_HH-MM'); % current date/time
            folder_name = [ 'rand-systems_' , dateString ];
            
            % create folder
            folder_name_full = [ 'datafiles' , filesep , folder_name ];
            mkdir( folder_name_full );
            
            % save data4sysid file for each system
            data4sysid_all = cell( size(data,2) , 1 );
            for i = 1 : size( data , 2 )
                data4sysid.folder_name = folder_name;
                data4sysid.train = cell(1, size(data,1)-1 );
                data4sysid.val = cell(1,1);
                for j = 1 : size( data , 1) - 1
                    data4sysid.train{j} = data{j,i};
                end
                data4sysid.val{1} = data{j+1,i};   % use last trial for validation
                
                % save the data4sysid file
                file_name = [ 'rsys-' , num2str(i) , '_' , 'train-' , num2str(j) , '_' , 'val-' , num2str(1) , '.mat' ];
                save( [ folder_name_full , filesep , file_name ] , '-struct' , 'data4sysid' );
                
                % save all systems in one big file
                data4sysid_all{i} = data4sysid;
            end
            
            % save the data4sysid file with all systems inside of it
            file_name_all = [ 'rsys-all' , '_' , 'train-' , num2str(j) , '_' , 'val-' , num2str(1) , '.mat' ];
            save( [ folder_name_full , filesep , file_name_all ] , 'data4sysid_all' );
        end
    end
end

