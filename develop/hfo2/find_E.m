function energy = find_E(box, r, type, q)

N = size(r, 1);
energy = 0;
alpha = 0.2;
rc = 10;
a = [0, 12372.16, 1388.77];
rho = [1, 0.2286, 0.3623];
c = [0, 81.34, 175.00];
v0=erfc(alpha*rc)/rc/rc + 2*alpha/sqrt(pi) * exp(-alpha*alpha*rc*rc)/rc;
selfE = 14.399645*sum(q.^2)* ...
    (alpha/sqrt(pi) ...
    + 0.5*erfc(alpha*rc)/rc ...
    + 0.5*(erfc(alpha*rc)/rc + 2*alpha/sqrt(pi) * exp(-alpha*alpha*rc*rc)));

for n1=1:N-1
    for n2=(n1+1):N
        r12=r(n2,:)-r(n1,:);                  % position difference vector
        r12=r12.';                            % column vector
        r12=box\r12;                          % transform to cubic box
        r12=r12-round(r12);                   % mininum image convention
        r12=box*r12;                          % transform back
        d12=sqrt(sum(r12.*r12));              % distance
        if d12 > rc
            continue;
        end
        t12=type(n1)+type(n2)-1;
        energy = energy + a(t12) * exp(-d12/rho(t12)) - c(t12) / (d12^6);
        energy = energy + 14.399645*q(n1)*q(n2)*(erfc(alpha*d12)/d12 - erfc(alpha*rc)/rc + v0*(d12-rc));
        if t12 == 2
            energy = energy + 0.3474 * (exp(-2*1.6230*(d12-2.0480)) - 2*exp(-1.6230*(d12-2.0480)));
        end
    end
end

energy = (energy-0*selfE)/N;

