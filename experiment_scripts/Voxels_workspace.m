%% REPRESENTATION OF A GRID
filename = 'data/grids/ur10e_grid2D_27.csv';

A = readmatrix(filename);
figure;
scatter(A(:,1),A(:,2),"Marker","o");
axis equal;
hold on;
plot(0,0,"Marker","x","Color",'r',MarkerSize=15)

%% REPRESENTATION OF REACHABILITY MAP

filename = 'data/eval_poses_ur10e/reachability_map_27_fused.csv';

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
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
% set(h, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha)    # activate only when checking, it takes a lot of resources

%% REPRESENTATION OF REACHABILITY MAP (ONLY FIRST QUADRANT: x >= 0, y >= 0)

filename = 'data/eval_poses_ur10e/reachability_map_27_fused.csv';

A = readmatrix(filename);

% Extract the first 375 rows (optional, if needed)
A_filtered = A(:, :);  % You can also just use A if no row filtering is needed

% Find the points where the 4th column is non-zero
non_zero_indices = A_filtered(:, 4) ~= 0;

% Apply additional condition: x >= 0 and y >= 0
x_positive = A_filtered(:, 1) >= 0;
y_positive = A_filtered(:, 2) >= 0;

% Combine all conditions
final_indices = non_zero_indices & x_positive & y_positive;

% Extract the corresponding points
x = A_filtered(final_indices, 1);
y = A_filtered(final_indices, 2);
z = A_filtered(final_indices, 3);
color = A_filtered(final_indices, 4);  % Color based on the 4th column

% Plot the points in 3D with color according to the 4th column
figure;
h = scatter3(x, y, z, 50, color, 'filled');  % 50 is the marker size
colormap('turbo(20)');
colorbar;
axis equal;
xlim([-1.3, 1.3])
ylim([-1.3, 1.3])
% zlim([-1.3, 1.3])
alpha = 0.5;
xlabel('X [m]');
ylabel('Y [m]');
zlabel('Z [m]');
% set(h, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha);  % Optional for transparency






