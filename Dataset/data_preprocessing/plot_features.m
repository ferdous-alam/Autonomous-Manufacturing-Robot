function plot_features(bandgap_properties, type)
 %% --------- plot figure ------------
 % ----- set latex font everywhere -------------
 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------

% color of the scatter points
scatter(bandgap_properties(:, 3), bandgap_properties(:, 4), 100, 'filled', 'MarkerFaceAlpha', 0.5);
box on; 
xlabel('peak frequencyc, $f_c$ (MHz)', 'FontSize', 18); 
ylabel('bandwidth, $w$(MHz)', 'FontSize', 18);

titlename = sprintf('%s', type); 
title(titlename, 'FontSize', 18); 

% ------ save plot ------
file_title = sprintf('Figures/feature_visualization_%s', type);  
% print(file_title, '-dpng', '-r300'); 

end
