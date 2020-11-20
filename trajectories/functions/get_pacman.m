function outline = get_pacman( center , radius )
%get_circle: get's coordinates of the outline of a circle
%   Detailed explanation goes here

t1 = (0 : 1/30 : 1)';
t2 = (pi/6 : pi/50 : 2*pi - pi/6)';
t3 = (0 : 1/30 : 1)';

%% top of mouth
outline = center + t1 .* [ radius*cos(pi/6) , radius*sin(pi/6) ]; 

%% body
outline = [ outline ;...
            radius * cos(t2) + center(1) , radius * sin(t2) + center(2) ];

%% bottom of mouth
outline = [ outline ;...
            ( center + [ radius*cos(-pi/6) , radius*sin(-pi/6) ] ) - t3 .* [ radius*cos(-pi/6) , radius*sin(-pi/6) ] ];


end