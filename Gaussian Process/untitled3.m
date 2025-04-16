clear all; close all;

data = load("cw1e.mat");
x = data.x;
y = data.y;

surf(reshape(x(:,1),11,11), reshape(x(:,2),11,11), reshape(y,11,11));
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Output');
title('Surface Plot of the Data');
