clear all;
close all;
clc;


load reaction_diffusion_data.mat;


% %%
jj = [150 160 170 180 190 200];
n = length(jj);
figure();
hold on;
for i = 1:6
    j = jj(i);
    subplot(2,n,i);
    pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
    subplot(2,n,n+i);
    pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
end

dt = t(2)-t(1);

%% Check truncation
u_data = reshape(u,[],length(t));
v_data = reshape(v,[],length(t));
X = [u_data;v_data];

[U,S,V] = svd(X,'econ');
figure();
hold on;
subplot(2,1,1);
plot(diag(S),'o');
xlim([0,70]);
subplot(2,1,2);
semilogy(diag(S),'o');
xlim([0,70]);

r = 10;

U = U(:,1:r);
V = V(:,1:r);
S = S(1:r,1:r);

for i = 1:size(X,2)
    temp = U'*X(:,i);
    X_cut(:,i) = U*temp;
end
u_cut = reshape(X_cut(1:size(u_data,1),:),size(u));
v_cut = reshape(X_cut(size(u_data,1)+1:end,:),size(u));

% jj = [150 160 170 180 190 200];
n = length(jj);
figure();
hold on;
for i = 1:6
    j = jj(i);
    subplot(2,n,i);
    pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
    subplot(2,n,n+i);
    pcolor(x,y,u_cut(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
end

figure();
hold on;
for i = 1:6
    j = jj(i);
    subplot(2,n,i);
    pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
    subplot(2,n,n+i);
    pcolor(x,y,v_cut(:,:,j)); shading interp; colormap(hot); colorbar;
    caxis([-1 1])
end

%% NN

V = U'*X;

aveV = mean(V,2);

input = V;
input = input(:,1:159)-aveV;
output = V;
output = output(:,2:160)-aveV;

% % net = feedforwardnet([20 10 20]);
% % net.layers{1}.transferFcn = 'poslin';
% % net.layers{2}.transferFcn = 'poslin';
% % net.layers{3}.transferFcn = 'poslin';
% % net = train(net,input,output);

numFeatures = 10;
numResponses = 10;
numHiddenUnits = 10;
 
layers = [ ...
 sequenceInputLayer(numFeatures)
 fullyConnectedLayer(20)
 fullyConnectedLayer(10)
 fullyConnectedLayer(20)
 fullyConnectedLayer(numResponses)
 regressionLayer];
 
options = trainingOptions('adam', ...
 'MaxEpochs',500, ...
 'GradientThreshold',1, ...
 'InitialLearnRate',0.005, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',125, ...
 'LearnRateDropFactor',0.2, ...
 'Verbose',0, ...
 'Plots','training-progress');
 
net = trainNetwork(input,output,layers,options);

predicted(:,1) = input(:,1);
for i = 2:201
    predicted(:,i) = predict(net,predicted(:,i-1));
end
predicted = U*(predicted+aveV);
u_pred = reshape(predicted(1:size(u_data,1),:),size(u));
v_pred = reshape(predicted(size(u_data,1)+1:end,:),size(u));

n = length(jj);
figure();
hold on;
for i = 1:6
    j = jj(i);
    subplot(2,n,i);
    pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
    xlabel('x')
    ylabel('y')
    title(['Time = ' num2str(t(j))])
    caxis([-1 1])
    subplot(2,n,n+i);
    pcolor(x,y,u_pred(:,:,j)); shading interp; colormap(hot); colorbar;
    xlabel('x')
    ylabel('y')
    title(['Time = ' num2str(t(j))])
    caxis([-1 1])
end

figure();
hold on;
for i = 1:6
    j = jj(i);
    subplot(2,n,i);
    pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
    xlabel('x')
    ylabel('y')
    title(['Time = ' num2str(t(j))])
    caxis([-1 1])
    subplot(2,n,n+i);
    pcolor(x,y,v_pred(:,:,j)); shading interp; colormap(hot); colorbar;
    xlabel('x')
    ylabel('y')
    title(['Time = ' num2str(t(j))])
    caxis([-1 1])
end