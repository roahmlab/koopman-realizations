function outline = get_circle( center , radius )
%get_circle: get's coordinates of the outline of a circle
%   Detailed explanation goes here

% t = (0 : pi/50 : 2*pi)';
t = (-pi/2 : pi/50 : 3*pi/2)';
outline = [ radius * cos(t) + center(1), radius * sin(t) + center(2)];


end

