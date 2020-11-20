% def_trajectory: Defines the refeference trajecory for the trial
%   Use this file to create custom reference trajectories.
%   On line 19, you call a function that generates a set of waypoints for a
%   given shape (examples of such functions are found in ./functions)


saveon = true; % should I save this model?

ref = struct;

ref.name = 'blockM_c0p45-0p35_0p5x0p5_45sec';

ref.T = 15;    % total time of trajectory (s)
ref.Ts = 0.05;  % length of timestep (s)

%% define shape of reference trajectory
addpath('functions');

% specify trajectory waypoints
y_old = get_blockM( [0.45,-0.35], 0.5 , 0.5 );   % collection of points that defines the shape of the trajectory (This works)

rmpath('functions');

%% flip sign of y-coordinate (for planar arm system)
y_old = [ y_old(:,1) , -y_old(:,2) ];

%% ensure trajectory starts from resting configuration of system
preamble = [ linspace( 0 , y_old(1,1) , 10 )' , linspace( 1 , y_old(1,2) , 10 )' ];
y_old = [ preamble(1:end-1,:) ; y_old ];   % planar manipulator

%% define time vector
t_old = linspace( 0 , ref.T , size( y_old , 1 ) )';
ref.t = ( 0 : ref.Ts : ref.T )'; % timestep must be the same as model.params.Ts

%% interpolate to match given timestep
ref.y = interp1( t_old , y_old , ref.t);

%% save reference trajectory struct
if saveon
    save(['files' , filesep , ref.name , '.mat'] , 'ref');
end