clear all; close all;

data = load("cw1e.mat");
x = data.x;
y = data.y;

scatter3(x(:,1), x(:,2), y, 'filled');
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Output');
title('3D Scatter Plot of the Data');
