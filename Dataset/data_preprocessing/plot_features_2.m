 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------


FEM = load('bandgap_properties_FEM_shortened.mat');
FEM_data = FEM.bandgap_properties_FEM;

exp = load('bandgap_properties_exp.mat');
exp_data = exp.bandgap_properties_exp; 
 
h = figure;
ax = gca; 
% color of the scatter points
scatter(FEM_data(:, 3), FEM_data(:, 4), 100, 'filled', 'MarkerFaceAlpha', 0.5);
hold on; 
scatter(exp_data(:, 3), exp_data(:, 4), 100, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('peak frequencyc, $f_c$ (MHz)', 'FontSize', 18); 
ylabel('bandwidth, $w$(MHz)', 'FontSize', 18);
box on; 
legend('FEM', 'DOE', 'FontSize', 14); 
hold off; 
ax.LineWidth = 1.5;

% remove extra white space in pdf 

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]);

file_title = sprintf('Figures/feature_visualization'); 
print(file_title, '-dpdf', '-r300'); 



