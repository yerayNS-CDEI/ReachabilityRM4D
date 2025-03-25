
filename = 'grid2D.csv';

A = readmatrix(filename);
figure;
plot(A(:,1),A(:,2),"Marker","o")
