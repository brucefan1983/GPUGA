function g=find_g_erhart(x)
gamma=0.09253;
c=1.13681;
d=0.63397;
h=-0.335;
g=1+c^2/d^2-c^2./(d^2+(x-h).^2);
g=g*gamma;
