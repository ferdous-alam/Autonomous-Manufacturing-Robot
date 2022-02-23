function plot_single_spectra(fft_data, f, dim, type)


 %% --------- plot figure ------------
 % ----- set latex font everywhere -------------
 set(groot,'defaultAxesTickLabelInterpreter','latex');   
 set(groot,'defaulttextinterpreter','latex');
 set(groot,'defaultLegendInterpreter','latex');
 % ----------------------------------------------

 % ----------------------------------------------
 figure(); 
 ax = gca; 
 set(gca, 'LooseInset', get(gca,'TightInset'))
 % plot contour of loss values
 plot(f, fft_data, 'LineWidth', 2.5); 
%  xline(desired(1) - desired(2), '--', 'color', 'black', 'LineWidth', 4.0); 
%  xline(desired(1) + desired(2), '--', 'color', 'black', 'LineWidth', 4.0); 
 xlabel('$f (MHz)$', 'FontSize', 14); 
 ylabel(' amplitude $(a.u.)$', 'FontSize', 14); 
 set(gcf, 'color', 'none');    
 set(gca, 'color', 'none');

%  titlename = sprintf('Spectra d%d lxy%d', dim(1), dim(2)); 
%  title(titlename, 'FontSize', 14); 
 ax.LineWidth = 2.0; 
 % ------ save plot ------
 file_title = sprintf('Figures/Spectra_visualization_%s_d%d_lxy%d.pdf', type, dim(1), dim(2));  
%  print(file_title, '-dpdf', '-r1200'); 
 exportgraphics(gcf,file_title, 'ContentType','vector', 'BackgroundColor','none')
end 