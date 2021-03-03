clear; close all;
r0 = [0, 0, 0;
      2, 0, 0];
box0 = [20, 0, 0;
    0, 1, 0;
    0, 0, 1];
N = size(r0, 1);
strain = -0.5:0.005:0.5;
N_strain = length(strain);

fid = fopen('train.in', 'w');
fprintf(fid,'%d %d\n',N_strain + N_strain, N_strain);
for n=1:(N_strain + N_strain)
    fprintf(fid,'%d\n',N);
end

energy=zeros(300,1);
virial=zeros(6,300);
for n = 1 : N_strain
    r = r0 * (1 + strain(n));
    box = box0 * (1 + strain(n));
    [energy(n), virial(:,n), force] = find_E(r, box);
    fprintf(fid,'%f %f %f %f %f %f %f %f %f\n',box);
    for i = 1 : N
        fprintf(fid,'0 %f %f %f %f %f %f\n',r(i,:), force(i,:));
    end
end

figure;
plot(energy(1:N_strain)/N, 'ro')
figure;
plot(virial(1, 1:N_strain)/N, 'd'); 


for n = 1 : N_strain
    r = r0 * (1 + strain(n));
    box = box0 * (1 + strain(n));
    [energy(n), virial(:, n), force] = find_E(r, box);
    fprintf(fid,'%f %f %f %f %f %f %f\n',energy(n), virial(:,n));
    fprintf(fid,'%f %f %f %f %f %f %f %f %f\n',box);
    for i = 1 : N
        fprintf(fid,'0 %f %f %f\n',r(i,:));
    end
end
fclose(fid);
