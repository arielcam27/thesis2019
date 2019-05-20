clear all;
close all;

m = 0;

x = linspace(0, 1.0, 500);
% t = linspace(0, 500.0, 2000);
t = linspace(0, 1000.0, 5000);
% t = linspace(0, 2000.0, 8000);

options=odeset('AbsTol',1e-10,'RelTol',1e-10);

sol = pdepe(m, @eqn, @initial, @bc, x, t, options);
u = sol(:,:,1);
v = sol(:,:,2);
w = sol(:,:,3);

figure;

subplot(1,3,1)
h = surf(x, t, u);
get(h);
set(h,'linestyle','none');
view(2);
title('Osteoclasts');
xlabel('Distance x');
ylabel('Time t');
colorbar;

subplot(1,3,2)
h = surf(x, t, v);
get(h);
set(h,'linestyle','none');
view(2);
title('Osteoblasts');
xlabel('Distance x');
ylabel('Time t');
colorbar;

subplot(1,3,3)
h = surf(x, t, w);
get(h);
set(h,'linestyle','none');
view(2);
title('Cancer cells');
ylabel('Time t');
colorbar;