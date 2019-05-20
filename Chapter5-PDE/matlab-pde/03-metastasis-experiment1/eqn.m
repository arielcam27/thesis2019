function [c,b,s] = eqn(x,t,u,DuDx)

[a1,a2,b1,b2,g1,g2,a3,b3,c1,c2,c3,c4,K] = pars();

DiffEpsi = 1e-6;
AdvEpsi  = 1e-4;

c = [1; 1; 1];

b = [0.5*DiffEpsi; 
     1.0*DiffEpsi;
     10.0*DiffEpsi] .* DuDx;
 
s = [a1*u(1)*(u(2)^g1) - b1*u(1) + c1*u(1)*u(3) - 1.0*AdvEpsi*DuDx(1); ...
     a2*(u(1)^g2)*u(2) - b2*u(2) + c2*u(2)*u(3) - 0.0*AdvEpsi*DuDx(2); ...
     a3*u(3)*(1.0-u(3)/K) - b3*u(3) + c3*(u(1)^g2)*u(3) + c4*(u(2)^g1)*u(3) - 0.0*AdvEpsi*DuDx(3)];

end