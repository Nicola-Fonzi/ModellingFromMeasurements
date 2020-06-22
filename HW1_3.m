%% Fitting of function
clear all
close all
clc

% Lotka-Volterra
% xdot = (b-py)x
% ydot = (rx-d)y

lepri = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18];
linci = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35];
year = 1845:2:1897;
lepri_test = [10 9 65];
linci_test = [12 12 25];
year_test = 1899:2:1903;
% 
year_interp = linspace(year(1),year(end),100);
x1 = spline(year,lepri,year_interp);
x2 = spline(year,linci,year_interp);
% year_interp = year;
% x1 = lepri;
% x2 = linci;

delta_t = year_interp(2)-year_interp(1);
lepri_dot(1) = (x1(2)-x1(1))/delta_t;
for i = 2:(length(x1)-1)
    lepri_dot(i) = (x1(i+1)-x1(i-1))/(2*delta_t);
end
lepri_dot(length(x1)) = (x1(end)-x1(end-1))/delta_t;

linci_dot(1) = (x2(2)-x2(1))/delta_t;
for i = 2:(length(x2)-1)
    linci_dot(i) = (x2(i+1)-x2(i-1))/(2*delta_t);
end
linci_dot(length(x2)) = (x2(end)-x2(end-1))/delta_t;

% f database x1^3 x2^3 x1 x2 x1x2
lepri_database = [x1(:) x1(:).*x2(:)];
linci_database = [x2(:) x1(:).*x2(:)];

csi_lepri = lepri_database\lepri_dot(:);
csi_linci = linci_database\linci_dot(:);

% We use euler to integrate
q(:,1) = [x1(1);x2(1)];
for i = 2:length(x1)
    q(1,i) = (csi_lepri(1)*q(1,i-1)+csi_lepri(2)*q(1,i-1)*q(2,i-1))*delta_t + q(1,i-1);
    q(2,i) = (csi_linci(1)*q(2,i-1)+csi_linci(2)*q(1,i-1)*q(2,i-1))*delta_t + q(2,i-1);
end

figure
subplot(1,2,1)
plot(year_interp,x1,'LineWidth',1.5)
hold on
plot(year_interp,q(1,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of hare individuals (thousands)')
legend('Real','L-V')
xlim([min(year) max(year_test)])
% ylim([0 160])

subplot(1,2,2)
plot(year_interp,x2,'LineWidth',1.5)
hold on
plot(year_interp,q(2,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of lynx individuals (thousands)')
legend('Real','L-V')
xlim([min(year) max(year_test)])
% ylim([0 160])

%% We use SINDY for better database

X_dot = [lepri_dot(:) linci_dot(:)];
f_database = [x1(:) x2(:) x1(:).*x2(:) ...
    x1(:).^2.*x2(:) x1(:).*x2(:).^2 x1(:).^2 ...
    x2(:).^2 cos(x1(:)) cos(x2(:)) 1./x1(:) 1./x2(:)];

csi(:,1) = lasso(f_database,X_dot(:,1),'Lambda',0.001);
csi(:,2) = lasso(f_database,X_dot(:,2),'Lambda',0.001);

[t,q]=ode45(@(t,q) integrationSINDY(t,q,csi), [1845 1897], [x1(1);x2(1)]);
t=t(:);
q=q';

figure
plot(year_interp,x1)
hold on
plot(t,q(1,:))

figure
plot(year_interp,x2)
hold on
plot(t,q(2,:))

labels = categorical({'x_1' 'x_2' 'x_1x_2' 'x_1^2x_2' 'x_1x_2^2' 'x_1^2' 'x_2^2'...
    'cos(x_1)' 'cos(x_2)' '1/x_1' '1/x_2'});
labels = reordercats(labels,{'x_1' 'x_2' 'x_1x_2' 'x_1^2x_2' 'x_1x_2^2' 'x_1^2' 'x_2^2' ...
    'cos(x_1)' 'cos(x_2)' '1/x_1' '1/x_2'});
figure
subplot(2,1,1)
bar(labels,csi(:,1))
ylabel('Loading for hares dynamics')

subplot(2,1,2)
bar(labels,csi(:,2))
ylabel('Loading for lynxes dynamics')


function [RHS]=integrationSINDY(t,q,csi)
RHS(1) = csi(1,1)*q(1)+csi(2,1)*q(2)+csi(3,1)*q(1)*q(2)...
    +csi(4,1)*q(1)^2*q(2)+csi(5,1)*q(1)*q(2)^2 ...
    +csi(6,1)*q(1)^2+csi(7,1)*q(2)^2+csi(8,1)*cos(q(1)) ...
    +csi(9,1)*cos(q(2))+csi(10,1)/q(1)+csi(11,1)/q(2);
RHS(2) = csi(1,2)*q(1)+csi(2,2)*q(2)+csi(3,2)*q(1)*q(2)...
    +csi(4,2)*q(1)^2*q(2)+csi(5,2)*q(1)*q(2)^2 ...
    +csi(6,2)*q(1)^2+csi(7,2)*q(2)^2+csi(8,2)*cos(q(1)) ...
    +csi(9,2)*cos(q(2))+csi(10,2)/q(1)+csi(11,2)/q(2);
RHS=RHS(:);
end