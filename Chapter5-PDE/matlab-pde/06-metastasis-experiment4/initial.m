function value = initial(x)

[a1,a2,b1,b2,g1,g2,a3,b3,c1,c2,c3,c4,K] = pars();

OCeq = (b2/a2)^(1/g2);
OBeq = (b1/a1)^(1/g1);

if x < 0.4 && x > 0.2
    OCinit = 10;
    OBinit = 5;
else
    OCinit = OCeq;
    OBinit = OBeq;
end

if x < 0.4 && x > 0.3
    CCinit = 1;
else
    CCinit = 0.0;
end

value = [OCinit; OBinit; CCinit];

end