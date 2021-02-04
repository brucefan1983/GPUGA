clear; close all;

% get the outputs from GPUGA:
f=cell(6,1);
load Si/force.out;f{1}=force;
load GaP/force.out;f{2}=force;
load GaAs/force.out;f{3}=force;
load InP/force.out;f{4}=force;
load InAs/force.out;f{5}=force;
load ZnSe/force.out;f{6}=force;


figure;
for n=1:6
    subplot(2,3,n);
    plot(f{n}(1:end,6),f{n}(1:end,3),'d');
    hold on;
    plot(f{n}(1:end,4),f{n}(1:end,1),'s');
    plot(f{n}(1:end,5),f{n}(1:end,2),'o');
    plot(-1:0.01:1,-1:0.01:1,'k--','linewidth',2);
    xlim([-1.1,1.1]);
    ylim([-1.1,1.1]);
    xlabel('Force (DFT) (eV/\AA)','fontsize',10,'interpreter','latex');
    ylabel('Force (Potential) (eV/\AA)','fontsize',10,'interpreter','latex');
    set(gca,'fontsize',10,'ticklength',get(gca,'ticklength')*2);
    if n==1
        legend('z','x','y');
        title('(a) Si');
    elseif n==2
        title('(b) GaP');
    elseif n==3
        title('(c) GaAs');
    elseif n==4
        title('(d) InP');
    elseif n==5
        title('(e) InAs');
    elseif n==6
        title('(f) ZnSe');
    end
end
