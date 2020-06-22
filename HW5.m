clear all; close all; clc

load BZ.mat
[m,n,k]=size(BZ_tensor); % x vs y vs time data

stampa = 0;

if stampa
    for j=1:k
        A=BZ_tensor(:,:,j);
        pcolor(A), shading interp, pause(0.2)
    end
end

%% SVD

X = reshape(BZ_tensor,[],size(BZ_tensor,3));
aveX = mean(X,2);
X = X - aveX;

trainLength = 999;
[U,S,V] = svd(X(:,1:trainLength),'econ');

figure();
hold on;
subplot(2,1,1);
plot(diag(S),'o');
subplot(2,1,2);
semilogy(diag(S),'o');

r = 40;

U = U(:,1:r);
V = V(:,1:r);
S = S(1:r,1:r);

for i = 1:size(X,2)
    temp = U'*X(:,i);
    X_cut(:,i) = U*temp + aveX;
end
BZ_cut = reshape(X_cut,size(BZ_tensor));

jj = [100 200 300 400 500 600];
n = length(jj);
figure();
hold on;
for i = 1:length(jj)
    j = jj(i);
    subplot(2,n,i);
    pcolor(BZ_tensor(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
    subplot(2,n,n+i);
    pcolor(BZ_cut(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
end


%% DMD

tempA = inv(S)*U';
tempA = V*tempA;
tempA = tempA*U;
A = U'*X(:,2:trainLength+1);
A = A*tempA;

XDMD(:,1) = U'*X(:,1);
for i = 2:size(X,2)
    XDMD(:,i)=A*XDMD(:,i-1);
end
XDMD = U*XDMD + aveX;

BZ_DMD = reshape(XDMD,size(BZ_tensor));

jj = [700 800 900 1000 1100 1200];
n = length(jj);
figure();
hold on;
for i = 1:length(jj)
    j = jj(i);
    subplot(2,n,i);
    pcolor(BZ_tensor(:,:,j)); shading interp; colormap(hot); colorbar;
%     caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
    subplot(2,n,n+i);
    pcolor(BZ_DMD(:,:,j)); shading interp; colormap(hot); colorbar;
%     caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
end


%% NN

numFeatures = size(X,1);
numResponses = size(X,1);
numHiddenUnits = r;

layers = [ ...
    sequenceInputLayer(numFeatures)
    fullyConnectedLayer(numHiddenUnits)
    eluLayer
    fullyConnectedLayer(numHiddenUnits)
    eluLayer
    fullyConnectedLayer(numHiddenUnits)
    eluLayer
    fullyConnectedLayer(numHiddenUnits)
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

trainLength = 299;
net = trainNetwork(X(:,1:trainLength),X(:,2:trainLength+1),layers,options);


Xpredicted(:,1)=X(:,1);
for i = 2:1200
    Xpredicted(:,i) = predict(net,Xpredicted(:,i-1));
end

Xpredicted = Xpredicted + aveX;
BZ_pred = reshape(Xpredicted,size(BZ_tensor));


jj = [100 200 300 400 500 600];
n = length(jj);
figure();
hold on;
for i = 1:length(jj)
    j = jj(i);
    subplot(2,n,i);
    pcolor(BZ_tensor(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
    subplot(2,n,n+i);
    pcolor(BZ_pred(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([0 140])
    xlabel('x')
    ylabel('y')
    title(['Time step = ' num2str(j)])
end