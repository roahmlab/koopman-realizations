function outline = get_blockM( center, width, height )
%setMtestPoints: Creates a file of test points in the shape of a Michigan M
%   Detailed explanation goes here

% set the increments between points
dw = width / 18;
dh = height / 11;

%% set outline points of M
outline = zeros(80,2);
outline(1,:) = center; % [center(1), 2*dh];
for i = 2:5
    outline(i,:) = outline(i-1,:) + [dw, dh];
end
for i = 6:10
    outline(i,:) = outline(i-1,:) + [dw, 0];
end
for i = 11:13
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
outline(14,:) = outline(13,:) + [-dw, 0];
for i = 15:19
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
outline(20,:) = outline(19,:) + [dw, 0];
for i = 21:23
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
for i = 21:23
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
for i = 24:29
    outline(i,:) = outline(i-1,:) + [-dw, 0];
end
for i = 30:32
    outline(i,:) = outline(i-1,:) + [0, dh];
end
outline(33,:) = outline(32,:) + [dw, 0];
for i = 34:37
    outline(i,:) = outline(i-1,:) + [0, dh];
end
for i = 38:41
    outline(i,:) = outline(i-1,:) + [-dw, -dh];
end
% left half
for i = 42:45
    outline(i,:) = outline(i-1,:) + [-dw, dh];
end
for i = 46:49
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
outline(50,:) = outline(49,:) + [dw, 0];
for i = 51:53
    outline(i,:) = outline(i-1,:) + [0, -dh];
end
for i = 54:59
    outline(i,:) = outline(i-1,:) + [-dw, 0];
end
for i = 60:62
    outline(i,:) = outline(i-1,:) + [0, dh];
end
outline(63,:) = outline(62,:) + [dw, 0];
for i = 64:68
    outline(i,:) = outline(i-1,:) + [0, dh];
end
outline(69,:) = outline(68,:) + [-dw, 0];
for i = 70:72
    outline(i,:) = outline(i-1,:) + [0, dh];
end
for i = 73:77
    outline(i,:) = outline(i-1,:) + [dw, 0];
end
for i = 78:81
    outline(i,:) = outline(i-1,:) + [dw, -dh];
end

end