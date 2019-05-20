function [pl,ql,pr,qr] = bc(xl,ul,xr,ur,t)

[a1,a2,b1,b2,g1,g2]=pars();

OCeq = (b2/a2)^(1/g2);
OBeq = (b1/a1)^(1/g1);

pl = [ul(1)-OCeq; ul(2)-OBeq];
ql = [0; 0];
pr = [ur(1)-OCeq; ur(2)-OBeq];
qr = [0; 0];

end