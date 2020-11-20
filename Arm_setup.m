% arm_setup.m
%
% Sets the values of parameters and derives equations of motion for a 
% hyper-redundant planar manipulator with the following properties:
%   -Every link has the same mass, inertia, length
%   -The input is groups of joint-torques
%   -Every module is identical (same number of joints/links)
%

saveon = true;  % decides whether to save the class or not

%% Define parameters
params = struct;

% params.sysName = 'thesis-arm-markers_grav-endload-01_3-mods_1-links_20hz';
params.sysName = 'single-pend2_1-mods_1-links_20hz';

params.Nmods = 1;   % number of modules (actuated sections)
params.nlinks = 1;      % number of links in each module
params.Nlinks = params.Nmods * params.nlinks;   % total number of links in robot

% general system parameters (make sure to include these an any system class)
params.nx = params.Nlinks * 2;   % dimension of the full state (joint angles and joing velocities)
params.ny = 2 * (params.Nlinks ); % + 2;   % dimension of measured output (mocap coordinates + end effector orientation)
params.nu = params.Nlinks;  % dimension of the input (reference angle at each joint)
params.nw = 2;  % dimension of load parametrization [end eff mass , gravity angle]

% manipulator parameters
params.L = 0.75; %0.3;    % total length of robot (m)
params.l = params.L / params.Nlinks;
% params.k = -0.00001;    % stiffness at each joint
params.k = -1e-5;    % stiffness at each joint
params.d = 1e1; %1e-4;    % viscous damping at each joint
params.m = 3e-1; %1e-1 , 0.0001;   % mass of each link (kg)
params.i = (1/3) * params.m * params.l^2;   % inertia of each link
params.g = 9.81;    % gravity constant (m/s^2)

% mocap parameters
params.markerPos = ( ( 0 : params.Nmods ) * params.l * params.nlinks ) / params.L;  % position of mocap markers along the arm

% input parameters
% params.ku = 1e-3; % effective input stiffness
params.ku = 1e1; % effective input stiffness

% simulation parameters
params.Ts = 0.05;   % sampling time (for 20 hz)
params.umax = 4*pi/8; % maximum input value (scalar for all modules, vector for different limits per module)


%% Create class for this system

Arm = Arm( params , 'output_type' , 'endeff');   % choice is 'angles' or 'markers' or 'endeff' or 'shape'

if saveon
    % save this system for later use
    dirname = [ 'systems' , filesep , params.sysName ];
    unique_dirname = auto_rename( dirname , '(0)' );
    Arm.params.sysName = erase( unique_dirname , ['systems', filesep] ) ;    % modify the system name parameter
    
    % create directory for the system, and save the class
    mkdir( unique_dirname );
    mkdir( [ unique_dirname , filesep , 'simulations' ] );  % make simulation subfolder
    fname = [ unique_dirname , filesep , params.sysName, '.mat' ];
    save( fname , 'Arm' );
end
