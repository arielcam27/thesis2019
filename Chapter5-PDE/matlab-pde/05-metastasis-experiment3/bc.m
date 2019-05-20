function [pl,ql,pr,qr] = bc(xl,ul,xr,ur,t)

[a1,a2,b1,b2,g1,g2,a3,b3,c1,c2,c3,c4,K]=pars();

OCeq = (b2/a2)^(1/g2);
OBeq = (b1/a1)^(1/g1);
CCeq = 0.0;

pl = [ul(1)-OCeq; ul(2)-OBeq; ul(3)-CCeq];
ql = [0; 0; 0];
pr = [ur(1)-OCeq; ur(2)-OBeq; ur(3)-CCeq];
qr = [0; 0; 0];

end