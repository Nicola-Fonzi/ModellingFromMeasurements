%% Homework 1

clear all
close all
clc

lepri = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18];
linci = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35];
year = 1845:2:1897;
lepri_test = [10 9 65];
linci_test = [12 12 25];
lepri2 = lepri.^2;
linci2 = linci.^2;
lincilepri = linci.*lepri;
lepri2_test = lepri_test.^2;
linci2_test = linci_test.^2;
lincilepri_test = linci_test.*lepri_test;
year_test = 1899:2:1903;
X = [lepri;linci;lepri2;linci2;lincilepri];
aveX = mean(X,2);
X = X-aveX;


%% Time embedding

timeEmb = 5;
clear H Hp
H=zeros(5*timeEmb,length(lepri)-timeEmb);
for i = 1:timeEmb
    H(i,:) = X(1,i:(end-(timeEmb-(i-1))));
end
for i = 1:timeEmb
    H(timeEmb+i,:) = X(2,i:(end-(timeEmb-(i-1))));
end
for i = 1:timeEmb
    H(2*timeEmb+i,:) = X(3,i:(end-(timeEmb-(i-1))));
end
for i = 1:timeEmb
    H(3*timeEmb+i,:) = X(4,i:(end-(timeEmb-(i-1))));
end
for i = 1:timeEmb
    H(4*timeEmb+i,:) = X(5,i:(end-(timeEmb-(i-1))));
end

Hp=zeros(5*timeEmb,length(lepri)-timeEmb);
for i = 1:timeEmb
    Hp(i,:) = X(1,(i+1):(end-(timeEmb-i)));
end
for i = 1:timeEmb
    Hp(timeEmb+i,:) = X(2,(i+1):(end-(timeEmb-i)));
end
for i = 1:timeEmb
    Hp(2*timeEmb+i,:) = X(3,(i+1):(end-(timeEmb-i)));
end
for i = 1:timeEmb
    Hp(3*timeEmb+i,:) = X(4,(i+1):(end-(timeEmb-i)));
end
for i = 1:timeEmb
    Hp(4*timeEmb+i,:) = X(5,(i+1):(end-(timeEmb-i)));
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
for i = 1:timeEmb
    q_initial=[q_initial ; X(3,i)];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(4,i)];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(5,i)];
end

q(:,1) = q_initial;
state(:,1) = q(:,1);

for i = 2:length(linci)
    q(:,i) = A*q(:,i-1);
    state(:,i) = q(:,i);
end

state(1,:) = state(1,:)+aveX(1);
state(timeEmb+1,:) = state(timeEmb+1,:)+aveX(2);
lepri_save = state(1,:);
linci_save = state(timeEmb+1,:);

figure
plot(year,lepri)
hold on
plot(year,state(1,:))

figure
plot(year,linci)
hold on
plot(year,state(timeEmb+1,:))

figure
plot(real(eig(A)),imag(eig(A)),'bo')
hold on
plot(cos(0:0.01:2*pi),sin(0:0.01:2*pi),'k--')

figure
plot(diag(S),'o')


clear q state q_initial
q_initial=[];
for i = 1:timeEmb
    q_initial=[q_initial ; X(1,end-(timeEmb-i))];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(2,end-(timeEmb-i))];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(3,end-(timeEmb-i))];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(4,end-(timeEmb-i))];
end
for i = 1:timeEmb
    q_initial=[q_initial ; X(5,end-(timeEmb-i))];
end
q(:,1) = q_initial;
state(:,1) = q(:,1);

for i = 2:(timeEmb+length(linci_test))
    q(:,i) = A*q(:,i-1);
    state(:,i) = q(:,i);
end

state(1,:) = state(1,:)+aveX(1);
state(timeEmb+1,:) = state(timeEmb+1,:)+aveX(2);
lepri_save = [lepri_save state(1,end-2:end)];
linci_save = [linci_save state(timeEmb+1,end-2:end)];

figure
plot([year(1,end-(timeEmb-1):end) year_test],[X(1,end-(timeEmb-1):end)+aveX(1) lepri_test])
hold on
plot([year(1,end-(timeEmb-1):end) year_test],state(1,:))

figure
plot([year(1,end-(timeEmb-1):end) year_test],[X(2,end-(timeEmb-1):end)+aveX(2) lepri_test])
hold on
plot([year(1,end-(timeEmb-1):end) year_test],state(timeEmb+1,:))

%% Final plot
figure
subplot(1,2,1)
plot([year year_test],[lepri lepri_test],'LineWidth',1.5)
hold on
plot([year year_test],lepri_save,'LineWidth',1.5)
xlabel('Year')
ylabel('Number of hare individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year_test)])
ylim([0 160])

subplot(1,2,2)
plot([year year_test],[linci linci_test],'LineWidth',1.5)
hold on
plot([year year_test],linci_save,'LineWidth',1.5)
xlabel('Year')
ylabel('Number of lynx individuals (thousands)')
legend('Real','DMD')
xlim([min(year) max(year_test)])
ylim([0 160])