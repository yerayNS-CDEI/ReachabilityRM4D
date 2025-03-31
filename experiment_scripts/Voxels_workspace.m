%% REPRESENTATION OF A GRID
filename = 'data/grids/ur10e_grid2D_9.csv';

A = readmatrix(filename);
figure;
scatter(A(:,1),A(:,2),"Marker","o");
axis equal;
hold on;
plot(0,0,"Marker","x","Color",'r',MarkerSize=15)

%% REPRESENTATION OF A GRID

filename = 'data/eval_poses_ur5e/reachability_map_22.csv';

A = readmatrix(filename);

% Assuming A is your matrix
A_filtered = A(:, :);  % Extract the first 375 rows

% Find the points where the 4th column is non-zero
non_zero_indices = A_filtered(:, 4) ~= 0;

% Extract the corresponding points
x = A_filtered(non_zero_indices, 1);
y = A_filtered(non_zero_indices, 2);
z = A_filtered(non_zero_indices, 3);
color = A_filtered(non_zero_indices, 4);  % Color based on the 4th column

% Plot the points in 3D with color according to the 4th column
figure;
h = scatter3(x, y, z, 50, color, 'filled');  % 50 is the marker size
colormap('turbo(20)');  % You can replace 'jet' with any other colormap like 'parula', 'cool', 'hot', etc.
colorbar;  % Optional: to show a color scale
axis equal
alpha = 0.5;
% set(h, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)    # activate just when checking, it takes a lot of resources

%% REPRESENTATION OF A GRID






