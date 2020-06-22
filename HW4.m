clear all, close all

% Simulate Lorenz system
dt = 0.01;
T = 10;
t = 0:dt:T;
nt = length(t);
b = 8/3;
sig = 10;
r_vect = [10, 28, 40];

Lorenz = @(t,x,r)([ sig * (x(2) - x(1))       ; ...
    x(1) * (r - x(3)) - x(2)  ; ...
    x(1) * x(2) - b * x(3)    ]);
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

%%
input=[]; output=[];
for i = 1:length(r_vect)
    figure
    r = r_vect(i);
    for j = 1:100  % training trajectories
        x0 = 30*(rand(3,1)-0.5);
        [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
        input=[input; [r*ones(nt-1,1), y(1:end-1,:)]];
        output=[output; y(2:end,:)];
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')
    end
end
grid on, view(-23,18)

%%
net = feedforwardnet([20 10 20]);
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,input.',output.');

% % % % numFeatures = 4;
% % % % numResponses = 3;
% % % %
% % % % layers = [ ...
% % % %  sequenceInputLayer(numFeatures)
% % % %  fullyConnectedLayer(20)
% % % %  reluLayer
% % % %  fullyConnectedLayer(10)
% % % %  reluLayer
% % % %  fullyConnectedLayer(20)
% % % %  reluLayer
% % % %  fullyConnectedLayer(numResponses)
% % % %  regressionLayer];
% % % %
% % % % options = trainingOptions('adam', ...
% % % %  'MaxEpochs',500, ...
% % % %  'GradientThreshold',1, ...
% % % %  'InitialLearnRate',0.005, ...
% % % %  'LearnRateSchedule','piecewise', ...
% % % %  'LearnRateDropPeriod',125, ...
% % % %  'LearnRateDropFactor',0.2, ...
% % % %  'Verbose',0, ...
% % % %  'Plots','training-progress');
% % % %
% % % % net = trainNetwork(input.',output.',layers,options);


%%

load netLorenz.mat

r_vect = [10, 28, 40, 17, 35];
for i = 1:5
    r = r_vect(i);
    figure
    x0=20*(rand(3,1)-0.5);
    [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
    xlabel('x')
    ylabel('y')
    zlabel('z')
    grid on
    
    ynn(1,:) = x0;
    for jj = 2:length(t)
        % % % %         y0 = predict(net,[r; x0]);
        y0 = net([r; x0]);
        ynn(jj,:) = y0.';
        x0 = y0;
    end
    plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
    
    figure
    subplot(3,1,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
    ylabel('x')
    subplot(3,1,2), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
    ylabel('y')
    subplot(3,1,3), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
    ylabel('z')
    xlabel('t')
end

%% Transition prediction

clear all; close all; clc

% Simulate Lorenz system
dt = 0.01;
T = 10;
t = 0:dt:T;
nt = length(t);
b = 8/3;
sig = 10;
r_vect = 28;

Lorenz = @(t,x,r)([ sig * (x(2) - x(1))       ; ...
    x(1) * (r - x(3)) - x(2)  ; ...
    x(1) * x(2) - b * x(3)    ]);
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
for i = 1:length(r_vect)
    figure
    r = r_vect(i);
    for j = 1:100  % training trajectories
        x0 = 30*(rand(3,1)-0.5);
        [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
        x = y(:,1);
        x1 = x;
        x1(x<0) = 0;
        x2 = x;
        x2(x>0) = 0;
        [aa1, bb1] = findpeaks(x1);
        [aa2, bb2] = findpeaks(-x2);
        aa2 = -aa2;
        [bb, ii] = sort([bb1;bb2]);
        aa = [aa1; aa2];
        aa = aa(ii);
        label = zeros(length(y),1);
        k = -1;
        j = 1;
        for i = 1:length(y)
            if bb(j) == i && j~=length(bb)
                if aa(j)*aa(j+1) < 0
                    k = -k;
                end
                j = j + 1;
            end
            label(i) = k;
        end
        input=[input; y(1:end-1,:)];
        output=[output; [y(2:end,:), label(2:end)]];
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')
        
    end
end
grid on, view(-23,18)

net = feedforwardnet([20 10 20]);
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net = train(net,input.',output.');
% save('netLorenz2.mat','net')

x0 = 30*(rand(3,1)-0.5);
[t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
figure
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
xlabel('x')
ylabel('y')
zlabel('z')
x = y(:,1);
        x1 = x;
        x1(x<0) = 0;
        x2 = x;
        x2(x>0) = 0;
        [aa1, bb1] = findpeaks(x1);
        [aa2, bb2] = findpeaks(-x2);
        aa2 = -aa2;
        [bb, ii] = sort([bb1;bb2]);
        aa = [aa1; aa2];
        aa = aa(ii);
        label = zeros(length(y),1);
        k = -1;
        j = 1;
        for i = 1:length(y)
            if bb(j) == i && j~=length(bb)
                if aa(j)*aa(j+1) < 0
                    k = -k;
                end
                j = j + 1;
            end
            label(i) = k;
        end
grid on
ynn(1,:) = x0;
for jj = 2:length(t)
    % % % %         y0 = predict(net,[r; x0]);
    out = net(x0);
    y0 = out(1:3);
    transition(jj) = out(4);
    ynn(jj,:) = y0.';
    x0 = y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
figure
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
ylabel('x')
xlabel('t')
title('State prediction')
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
ylabel('y')
xlabel('t')
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
ylabel('z')
xlabel('t')
subplot(3,2,[2 4 6])
plot(t,label,'Linewidth',2)
hold on
plot(t,transition,'Linewidth',2)
xlabel('time')
title('Transition prediction')
ylabel('Flag value')