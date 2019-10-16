function g=find_g_tersoff(x)
gamma=1.1000e-6;
c=1.0039e5;
d=16.217;
h=-0.59825;
g=1+c^2/d^2-c^2./(d^2+(x-h).^2);
g=g*gamma;
