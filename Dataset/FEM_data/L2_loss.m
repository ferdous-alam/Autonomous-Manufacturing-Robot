% Md Ferdous Alam, 05/27/2021
% HRL, OSU 
% Title: L2 loss for comparing spectra 

%% desired spectra
x = [0:2/160:2];
y = normpdf(x,0.63,0.15/2);
y = y /max(y); 
% load data 
data = load('data\Zhi_TransmissionSpectra-L4000-Final.mat');
zhi_loss_data = readtable('data\TotoalLoss_1_Pair1_L4000_07152020.csv');


% extract field names from x.mat file
transmission_spectra = data.TransmissionSpectra; 
names = fieldnames(transmission_spectra);
x_axis = transmission_spectra.(names{1})(1:161)/1e6;

L2_loss_vals = zeros(48, 1); 
KL_loss_vals = zeros(48, 1);
    
h1 = figure(1);
for k=2:length(names)
    sample1 = transmission_spectra.(names{k}).Sample1;
    sample2 = transmission_spectra.(names{k}).Sample2;
    sample3 = transmission_spectra.(names{k}).Sample3;
    freq1 = sample1(1:161);
    freq2 = sample2(1:161);
    freq3 = sample3(1:161);
    avg_freq = ((freq1+freq2+freq3)/3);
    avg_freq = avg_freq/max(avg_freq); 
    
    % calculate L2 loss
    L2_loss_vals(k-1) = norm(y - avg_freq, 2); 
    
    % KL-divergence loss
    y_t = reshape(y, [161, 1]); 
    KL_loss_vals(k-1) = sum(avg_freq.*log(y_t./avg_freq)); 
    
    subplot(6, 8, k-1); 
    plot(x, y);
    hold on; 
    plot(x_axis, avg_freq); 
    title(['L = ', num2str(KL_loss_vals(k-1))],  'Fontsize', 4); 
    hold off; 
    text(1.0, 0.7,sprintf('%s', names{k}), 'fontsize', 2); 
    set(gca,'FontSize',4)

%     pause(0.5);
    
end

set(h1,'Units','Inches');
pos = get(h1,'Position');
set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h1,'experimental-spectra-and-loss_zhi-data','-dpdf','-r0')


h2 = figure(2);
loss_vals = reshape(KL_loss_vals, [8, 6]); 
x = 3.5:0.5:6;
y = 7:0.5:10.5;
[X, Y] = meshgrid(x, y);
subplot(2,1, 1)
surf(X, Y, loss_vals);
title('L2 loss')
colorbar;
set(h2,'Units','Inches');
pos = get(h2,'Position');
set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

lxy = zhi_loss_data{:, 1}; 
dia = zhi_loss_data{:, 2};
zhi_loss = zhi_loss_data{:, 3};
loss_sample1 = zhi_loss(1:48); 
loss_sample2 = zhi_loss(49:96); 
loss_sample3 = zhi_loss(97:144); 
zhi_average_loss = (1/3)*(loss_sample1 + loss_sample2 +loss_sample3);
zhi_loss_vals = reshape(zhi_average_loss, [8, 6]);
subplot(2,1, 2)
surf(X, Y, zhi_loss_vals);
title('zhi loss')
colorbar; 

print(h2,'experimental-loss-surface_zhi-data','-dpdf','-r0');


h3 = figure(3);
subplot(2,1, 1);
contourf(X, Y, loss_vals);
title('KLD loss')
colorbar;
subplot(2,1, 2)
contourf(X, Y, zhi_loss_vals);
title('zhi loss')
colorbar; 

set(h3,'Units','Inches');
pos = get(h3,'Position');
set(h3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h3,'experimental-loss-contour_zhi-data','-dpdf','-r0');

