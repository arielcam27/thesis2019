clear all;
close all;

m = 0;

x = linspace(0, 1.0, 500);
% t = linspace(0, 500.0, 2000);
% t = linspace(0, 1000.0, 5000);
t = linspace(0, 2000.0, 8000);

options=odeset('NonNegative',[]);

sol = pdepe(m, @eqn, @initial, @bc, x, t, options);
u = sol(:,:,1);
v = sol(:,:,2);

figure;

subplot(2,2,1)
h = surf(x, t, u);
get(h);
set(h,'linestyle','none');
view(2);
title('Osteoclasts');
xlabel('Distance x');
ylabel('Time t');

subplot(2,2,2)
h = surf(x, t, v);
get(h);
set(h,'linestyle','none');
view(2);
title('Osteoblasts');
subplot(2,2,1)
ylabel('Time t');

subplot(2,2,3)
hold on;
plot(x, u(1,:));
plot(x, u(floor(length(t)/2),:));
plot(x, u(length(t),:));

subplot(2,2,4)
hold on;
plot(x, v(1,:));
plot(x, v(floor(length(t)/2),:));
plot(x, v(length(t),:));