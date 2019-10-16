function g=find_g_schall(x)
x_special=[1,-1/3,-1/2,-1];
g_special=[0.98,-0.000223579,-0.044734259,-0.122736935];
g=spline(x_special,g_special,x);