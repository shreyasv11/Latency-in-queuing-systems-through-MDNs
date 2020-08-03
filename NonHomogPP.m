function [ti] = NonHomogPP(endTime, avgArrRate, relAmp, cycle)

lambda = @(x) avgArrRate*(1 + relAmp*sin((2*pi*x)/cycle));
maxl = lambda(fminbnd(@(x) -lambda(x),0,endTime));

ti = [];
t = 0;
i = 1;
while t<endTime
    u1 = rand;
    t = t - (1/maxl)*(log(u1));
    u2 = rand;
    if u2 <= lambda(t)/maxl
        ti(i) = t;
        i = i + 1;
    end
end
ti=ti(ti<endTime);
end