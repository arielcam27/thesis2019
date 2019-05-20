function value = initial(x)

[a1,a2,b1,b2,g1,g2] = pars();

OCeq = (b2/a2)^(1/g2);
OBeq = (b1/a1)^(1/g1);

if x < 0.5 && x > 0.1
    value = [10*OCeq; 0.1*OBeq];
else
    value = [OCeq; OBeq];
end

end