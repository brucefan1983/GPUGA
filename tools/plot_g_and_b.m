clear;close all;

theta=0:0.05:pi;
x=cos(theta);
theta=theta*180/pi;

g_tersoff=find_g_tersoff(x);
g_erhart=find_g_erhart(x);
g_kumagai=find_g_kumagai(x);
g_pun=find_g_pun(x);
g_mini=find_g_mini(x);
g_schall=find_g_schall(x);

b_tersoff=(1+g_tersoff.^(0.78734)).^(-1/(2*0.78734));
b_erhart=(1+g_erhart.^1).^(-1/(2*1));
b_kumagai=(1+g_kumagai.^1).^(-0.53298909);
b_pun=(1+g_pun.^(2.16152496)).^(-0.544097766/2.16152496);
b_mini=(1+g_mini.^(0.602568)).^(-1/(2*0.602568));  
b_schall=(1+g_schall.^1).^(-1/(2*1));

figure;
subplot(2,1,1);
plot(theta,g_tersoff,'d','linewidth',1);
hold on;
plot(theta,g_erhart,'s','linewidth',1);
plot(theta,g_kumagai,'o','linewidth',1);
plot(theta,g_pun,'<','linewidth',1);
plot(theta,g_schall,'>','linewidth',1);
plot(theta,g_mini,'*','linewidth',1);
legend('Tersoff-1989','Erhart-2005','Kumagai-2007','Pun-2017','Schall-2008','mini-Tersoff');
set(gca,'fontsize',12,'xtick',0:30:180);
xlim([0,180]);
ylim([-0.2,1.6]);
xlabel('$\theta$ (degree)','fontsize',12,'interpreter','latex');
ylabel('$g(\theta)$','fontsize',12,'interpreter','latex');
set(gca,'ticklength',get(gca,'ticklength')*2);
text(-20,1.7, '(a)','fontsize',12);

subplot(2,1,2);
plot(theta,b_tersoff,'d','linewidth',1);
hold on;
plot(theta,b_erhart,'s','linewidth',1);
plot(theta,b_kumagai,'o','linewidth',1);
plot(theta,b_pun,'<','linewidth',1);
plot(theta,b_schall,'>','linewidth',1);
plot(theta,b_mini,'*','linewidth',1);
set(gca,'fontsize',12,'xtick',0:30:180);
xlim([0,180]);
ylim([0.57,1.1]);
xlabel('$\theta$ (degree)','fontsize',12,'interpreter','latex');
ylabel('$b(\theta)$','fontsize',12,'interpreter','latex');
set(gca,'ticklength',get(gca,'ticklength')*2);
text(-20,1.15, '(b)','fontsize',12);
