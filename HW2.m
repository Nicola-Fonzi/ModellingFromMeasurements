clear all; close all; clc

%% Parameters

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs
usave=[];
usavep=[];
for index = 1:1000

stampa = 0;
N = 2^8;
x = 2*pi*(1:N)'/N;
u=randn(N,1);
v = fft(u);
nu=0.005;

%% Spatial grid and initial condition:
h = 0.01;
k = [0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - nu*k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

%% Main time-stepping loop:
uu = u; tt = 0;
tmax = 10; nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);
uu = zeros(N,nmax);

for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2);
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u;
        tt(n) = t;
    end
end


%% Plot pcolor
cutoff = tt > 0;
cutoff = cutoff & tt<4;
tsave = tt(1:399);
usave = [usave uu(:,1:399)];
if index == 1
    uplot = uu(:,1:599);
end
usavep = [usavep uu(:,2:400)];
xsave = x/(2*pi);
dt = h;
dx = 1/N;

if stampa == 1
    pcolor(x,tt(cutoff),uu(:,cutoff).'),shading interp, colormap(hot)
end
%% Further visualisation
if stampa == 1
    uu2=uu(:,cutoff); [mm,nn]=size(uu2);
    for j=1:nn
        uut(:,j)=abs(fftshift(fft(uu2(:,j))));
    end
    k=[0:N/2-1 -N/2:-1].';
    ks=fftshift(k);
    figure(4)
    surfl(ks,tt(cutoff),(uut(:,cutoff).')),shading interp, colormap(hot), %view(15,50)
    figure(5)
    waterfall(ks,tt(cutoff),(uut(:,cutoff).')), colormap([0 0 0])
    set(gca,'Xlim',[-50 50])
end

end
%% Train

% % % % % % net = feedforwardnet([400 300 400]);
% % % % % % net.trainFcn='traingdm';
% % % % % % net.trainParam.lr = 1e-2;
% % % % % % net.trainParam.max_fail = 100;
% % % % % % net.trainParam.epochs = 1000000;
% % % % % % net.divideFcn = 'dividetrain';
% % % % % % net.performParam.normalization = 'percent';
% % % % % % % net.layers{1}.transferFcn = 'tansig';
% % % % % % % net.layers{2}.transferFcn = 'tansig';
% % % % % % % net.layers{3}.transferFcn = 'tansig';
% % % % % % net = train(net,usave,usavep);

numFeatures = 256;
numResponses = 256;
numHiddenUnits = 1000;

layers = [ ...
    sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits)
    reluLayer
%     tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
%     tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
%     tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    reluLayer
%     tanhLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');


net = trainNetwork(usave,usavep,layers,options);


% % % % net = predictAndUpdateState(net,usave);
% % % % usave(:,400) = usavep(:,end);
% % % % [net,usave(:,401)] = predictAndUpdateState(net,usavep(:,end));
% % % % 
% % % % for i = 402:599
% % % %     [net,usave(:,i)] = predictAndUpdateState(net,usave(:,i-1),'ExecutionEnvironment','cpu');
% % % % end

upredicted(:,1)=uplot(:,1);
for i = 2:599
upredicted(:,i) = predict(net,upredicted(:,i-1));
end


%% Plot trained

pcolor(x,tt(1:599),upredicted.'),shading interp, colormap(hot)
figure
pcolor(x,tt(1:599),uplot.'),shading interp, colormap(hot)