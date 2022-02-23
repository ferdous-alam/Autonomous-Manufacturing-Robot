function visualize_single_spectra(spectra, f, dim)


 %% --------- plot figure ------------
 % ----- set latex font everywhere -------------
 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------

 % ----------------------------------------------
 h = figure;
 ax = gca; 
 % plot contour of loss values
 plot(f, spectra, 'color', 'red', 'LineWidth', 3.5); 
 box on; 
 ax.LineWidth = 2.5; 
 xlabel('$f (MHz)$', 'FontSize', 14); 
 ylabel(' amplitude $(a.u.)$', 'FontSize', 14); 

 % ------ save plot ------
 file_title = sprintf('Figures/Spectra_visualization_d%d_lxy%d', dim(1), dim(2));  

 % remove extra white space in pdf 
 set(h,'Units','Inches');
 pos = get(h,'Position');
 set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3),pos(4)]); 
 print(file_title, '-dpdf', '-r300'); 

end 