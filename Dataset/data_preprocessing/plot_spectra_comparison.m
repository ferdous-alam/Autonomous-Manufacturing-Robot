function plot_spectra_comparison(fft_data_fem, f_fem, fft_data_exp, f_exp, dim)


 %% --------- plot figure ------------
 % ----- set latex font everywhere -------------
 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------

 % ----------------------------------------------
 figure(); 
 ax = gca; 
 % plot contour of loss values
 plot(f_fem, fft_data_fem, 'LineWidth', 1.5); 
 hold on; 
 plot(f_exp, fft_data_exp, 'LineWidth', 1.5); 
%  xline(desired(1) - desired(2), '--', 'linewidth', 1.5); 
%  xline(desired(1) + desired(2), '--', 'linewidth', 1.5); 
 hold off; 
 legend('FEM', 'Experimental'); 
 xlabel('$f (MHz)$', 'FontSize', 14); 
 ylabel(' amplitude $(a.u.)$', 'FontSize', 14); 
 titlename = sprintf('Spectra d%d lxy%d', dim(1), dim(2)); 
 title(titlename, 'FontSize', 14); 

 % ------ save plot ------
 file_title = sprintf('Figures/Spectra_visualization_d%d_lxy%d', dim(1), dim(2));  
 print(file_title, '-dpng', '-r300'); 

end 