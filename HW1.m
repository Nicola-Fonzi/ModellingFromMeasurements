%% Homework 1

clear all
close all
clc

lepri = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18];
linci = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35];
year = 1845:2:1897;
lepri_test = [10 9 65];
linci_test = [12 12 25];
year_test = 1899:2:1903;
X = [lepri;linci];
aveX = mean(X,2);
X = X-aveX;

figure
plot(year,lepri,'LineWidth',1.5)
hold on
plot(year,linci,'LineWidth',1.5)
xlabel('Year')
ylabel('Number of individuals (thousands)')
legend('Hares','Lynxes')
xlim([min(year) max(year)])
ylim([0 160])

H = [lepri(1:end-1);linci(1:end-1)];
Hp = [lepri(2:end);linci(2:end)];

[U,S,V] = svd(H,'econ');

A=Hp*V*inv(S)*U';

q(:,1) = [lepri(1);linci(1)];
state(:,1) = q(:,1);

for i = 2:length(linci)
    q(:,i) = A*q(:,i-1);
    state(:,i) = q(:,i);
end


figure
subplot(1,2,1)
plot(year,lepri,'LineWidth',1.5)
hold on
plot(year,state(1,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of hare individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year)])
ylim([0 160])

subplot(1,2,2)
plot(year,linci,'LineWidth',1.5)
hold on
plot(year,state(2,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of lynx individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year)])
ylim([0 160])

%% Time embedding
clc
timeEmb = 5;
clear H Hp
H=zeros(2*timeEmb,length(lepri)-timeEmb);
for i = 1:timeEmb
    H(i,:) = X(1,i:(end-(timeEmb-(i-1))));
end
for i = 1:timeEmb
    H(timeEmb+i,:) = X(2,i:(end-(timeEmb-(i-1))));
end

Hp=zeros(2*timeEmb,length(lepri)-timeEmb);
for i = 1:timeEmb
    Hp(i,:) = X(1,(i+1):(end-(timeEmb-i)));
end
for i = 1:timeEmb
    Hp(timeEmb+i,:) = X(2,(i+1):(end-(timeEmb-i)));
end

[U,S,V] = svd(H,'econ');

A=Hp*V*inv(S)*U';

clear q state q_initial
q_initial=[];
for i = 1:timeEmb
    q_initial=[q_initial ; X(1,i)];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(2,i)];
end
q(:,1) = q_initial;
state(:,1) = q(:,1);

for i = 2:length(linci)
    q(:,i) = A*q(:,i-1);
    state(:,i) = q(:,i);
end

state(1,:) = state(1,:)+aveX(1);
state(timeEmb+1,:) = state(timeEmb+1,:)+aveX(2);

figure
subplot(1,2,1)
plot(year,lepri,'LineWidth',1.5)
hold on
plot(year,state(1,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of hare individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year)])
ylim([0 160])

subplot(1,2,2)
plot(year,linci,'LineWidth',1.5)
hold on
plot(year,state(timeEmb+1,:),'LineWidth',1.5)
xlabel('Year')
ylabel('Number of lynx individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year)])
ylim([0 160])

figure
plot(real(eig(A)),imag(eig(A)),'bo')
hold on
plot(cos(0:0.01:2*pi),sin(0:0.01:2*pi),'k--')

figure
plot(diag(S),'o','LineWidth',1.5)
ylabel('Singular value of the snapshot matrix')


clear q state q_initial
q_initial=[];
for i = 1:timeEmb
    q_initial=[q_initial ; X(1,end-(timeEmb-i))];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(2,end-(timeEmb-i))];
end
q(:,1) = q_initial;
state(:,1) = q(:,1);

for i = 2:(timeEmb+length(linci_test))
    q(:,i) = A*q(:,i-1);
    state(:,i) = q(:,i);
end

state(1,:) = state(1,:)+aveX(1);
state(timeEmb+1,:) = state(timeEmb+1,:)+aveX(2);

figure
plot([year(1,end-(timeEmb-1):end) year_test],[X(1,end-(timeEmb-1):end)+aveX(1) lepri_test])
hold on
plot([year(1,end-(timeEmb-1):end) year_test],state(1,:))