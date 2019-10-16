clear;close all;
load a.txt; % DFT input
load prediction.out; % GPUGA output
load force.out; % GPUGA output

prediction=prediction/64;
N=53;
M=4;
energy=prediction(M+1:N*3+M,:);
virial=prediction(N*3+M+1:end,:);

df=force(:,1:3)-force(:,4:6);
df=reshape(df,960,1);
f=reshape(force(:,4:6),960,1);
error_f=std(df,1)/std(f,1)
de=prediction(5:end/4,1)-prediction(5:end/4,2);
e=prediction(5:end/4,2);
error_e=std(de,1)/std(e,1)
dv=prediction(end/4+1:end,1)-prediction(end/4+1:end,2);
v=prediction(end/4+1:end,2);
error_v=std(dv,1)/std(v,1)

figure;

subplot(1,3,1)
plot(force(1:end,4:6),force(1:end,1:3),'o');
xlim([-2,2]);
ylim([-2,2]);
xlabel('Force (DFT) (eV/\AA)','fontsize',12,'interpreter','latex');
ylabel('Force (Potential) (eV/\AA)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
title('(a)');

subplot(1,3,2);
n=1;
index=(n-1)*N+1:n*N;
plot(a,energy(index,2),'d');
hold on;
plot(a,energy(index,1),'-','linewidth',2);
n=2;
index=(n-1)*N+1:n*N;
plot(a,energy(index,2),'s');
plot(a,energy(index,1),'--','linewidth',2);
n=3;
index=(n-1)*N+1:n*N;
plot(a,energy(index,2),'o');
plot(a,energy(index,1),'-.','linewidth',2);
axis tight
xlabel('$a$ (\AA)','fontsize',12,'interpreter','latex');
ylabel('Energy (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
legend('triaxial-DFT','triaxial-Potential','biaxial-DFT',...
   'biaxial-Potential','uniaxial-DFT','uniaxial-Potential');
title('(b)');

subplot(1,3,3);
plot(a,virial(M+1:N+M,2),'d','linewidth',1);
hold on;
plot(a,virial(M+1:N+M,1),'-','linewidth',2);
plot(a,virial(N+M+1:2*N+M,2),'d','linewidth',1);
plot(a,virial(N+M+1:2*N+M,1),'-','linewidth',2);
plot(a,virial(end-N+1:end,2),'s','linewidth',1);
plot(a,virial(end-N+1:end,1),'--','linewidth',2);
axis tight
xlabel('$a$ (\AA)','fontsize',12,'interpreter','latex');
ylabel('Virial (eV)','fontsize',12,'interpreter','latex');
set(gca,'fontsize',12,'ticklength',get(gca,'ticklength')*2);
title('(c)');
