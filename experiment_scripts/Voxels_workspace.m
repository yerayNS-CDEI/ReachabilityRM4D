
filename = 'data/grids/grid2D_11.csv';

A = readmatrix(filename);
figure;
plot(A(:,1),A(:,2),"Marker","o")
