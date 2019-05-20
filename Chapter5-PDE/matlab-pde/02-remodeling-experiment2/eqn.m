function [c,b,s] = eqn(x,t,u,DuDx)

[a1,a2,b1,b2,g1,g2] = pars();

DiffEpsi = 1e-6;
AdvEpsi  = 1e-4;

c = [1; 1];
b = [1*DiffEpsi; 2.0*DiffEpsi] .* DuDx;
s = [a1*u(1)*(u(2)^g1)-b1*u(1)-1.5*AdvEpsi*DuDx(1); 
     a2*(u(1)^g2)*u(2)-b2*u(2)-0.0*AdvEpsi*DuDx(2)];

end