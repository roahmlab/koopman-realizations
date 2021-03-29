% example_control.m
% 
% This script constructs a linear, bilinear, and nonlinear MPC
% controller based on included model realizations. It will simulate the a
% trajectory following task using each controller and plot the results. You can
% change the parameters of the MPC controllers by modifying the Name/Value
% arguments into Kmpc each time it is called.
%
% NOTE: Before running this scripte you need to run 'example_sysid.m' first 
% so that Ksysid_linear, Ksysid_bilinear and Ksysid_nonlinear are in the 
% workspace.

%% load in arm model
load([ 'systems' , filesep , 'thesis-arm-markers_noload_3-mods_1-links_20hz' , filesep , 'thesis-arm-markers_noload_3-mods_1-links_20hz.mat' ]);


%% create mpc controller classes

Kmpc_linear = Kmpc( Ksysid_linear ,...
        'horizon' , 10 ,...
        'input_bounds' , [ -7*pi/8 , 7*pi/8 ] ,...
        'input_slopeConst' , 1e-1 ,...
        'input_smoothConst' , [] ,... % [1e-1] ,...
        'state_bounds' , [] ,...
        'cost_running' , 10 ,...   % 0.1
        'cost_terminal' , 100 ,...  % 100
        'cost_input' , 0.1 * [ 3e-2 , 2e-2 , 1e-2 ]' ,...    % 1e-1
        'projmtx' , Ksysid_linear.model.C(end-1:end,:) ); %,...  % just end effector


Kmpc_bilinear = Kmpc( Ksysid_bilinear ,...
        'horizon' , 10 ,...
        'input_bounds' , [ -7*pi/8 , 7*pi/8 ] ,...
        'input_slopeConst' , 1e-1 ,...
        'input_smoothConst' , [] ,... % [1e-1] ,...
        'state_bounds' , [] ,...
        'cost_running' , 10 ,...   % 0.1
        'cost_terminal' , 100 ,...  % 100
        'cost_input' , 0.1 * [ 3e-2 , 2e-2 , 1e-2 ]' ,...    % 1e-1
        'projmtx' , Ksysid_bilinear.model.C(end-1:end,:) ); %,...  % just end effector


Kmpc_nonlinear = Kmpc( Ksysid_nonlinear ,...
        'horizon' , 10 ,...
        'input_bounds' , [ -7*pi/8 , 7*pi/8 ] ,...
        'input_slopeConst' , 1e-1 ,...
        'input_smoothConst' , [] ,... % [1e-1] ,...
        'state_bounds' , [] ,...
        'cost_running' , 10 ,...   % 0.1
        'cost_terminal' , 100 ,...  % 100
        'cost_input' , 0.1 * [ 3e-2 , 2e-2 , 1e-2 ]' ,...    % 1e-1
        'projmtx' , Ksysid_nonlinear.model.C(end-1:end,:) ,...  % just end effector
        'mpc_type' , 'nonlinear' );  % only need specify if you want to do nonlinear mpc with a bilinear model
    
%% create simulation classes

Ksim_linear = Ksim( Arm , Kmpc_linear );
Ksim_bilinear = Ksim( Arm , Kmpc_bilinear );
Ksim_nonlinear = Ksim( Arm , Kmpc_nonlinear );

%% run controller simulations

% load in reference trajectory
load([ 'trajectories' , filesep , 'files' , filesep , 'blockM_c0p45-0p35_0p5x0p5_15sec.mat' ]);

% simulate controllers
results_linear = Ksim_linear.run_trial_mpc( ref.y , [] , [] );
results_bilinear = Ksim_bilinear.run_trial_mpc( ref.y , [] , [] );
results_nonlinear = Ksim_nonlinear.run_trial_mpc( ref.y , [] , [] );

%% plot results

figure;
subplot(1,3,1)  % linear controller results
title('Linear');
hold on;
plot( ref.y(:,1) , ref.y(:,2) );
plot( results_linear.Y(:,5) , results_linear.Y(:,6) );
hold off;
set(gca, 'YDir','reverse');
ylim([0,1]);
xlim([0,1]);
box on; grid on;
legend('Reference' , 'K-MPC Controller' , 'Location' , 'southeast');

subplot(1,3,2)  % bilinear controller results
title('Bilinear');
hold on;
plot( ref.y(:,1) , ref.y(:,2) );
plot( results_bilinear.Y(:,5) , results_bilinear.Y(:,6) );
hold off;
set(gca, 'YDir','reverse');
ylim([0,1]);
xlim([0,1]);
box on; grid on;
legend('Reference' , 'K-BMPC Controller' , 'Location' , 'southeast');

subplot(1,3,3)  % nonlinear controller results
title('Nonlinear')
hold on;
plot( ref.y(:,1) , ref.y(:,2) );
plot( results_nonlinear.Y(:,5) , results_nonlinear.Y(:,6) );
hold off;
set(gca, 'YDir','reverse');
ylim([0,1]);
xlim([0,1]);
box on; grid on;
legend({'Reference' , 'K-NMPC Controller'} , 'Location' , 'southeast');










