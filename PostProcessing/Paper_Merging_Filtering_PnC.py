import os
import mat4py
from mat4py import savemat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import math
from numpy import genfromtxt
'''
need only frequency domain!!
'''
# This code is used to generate merged reference signal from multiple experimental data.

plt.close('all')

# Preparation

# log-based sequence
starting_freq = math.log2(1e6)-4*(math.log2(4e6)-math.log2(1e6))/4
incrementing_freq = (math.log2(4e6)-math.log2(1e6))/4
ending_freq = math.log2(4e6)+(math.log2(4e6)-math.log2(1e6))/4

freq_updated = starting_freq
freq_full_range = []
while freq_updated <= ending_freq:
    freq_full_range.append(2**freq_updated)
    freq_updated += incrementing_freq

# freq_breaks for freq localizing basis functions
ending_freq = math.log2(4e6)+2*(math.log2(4e6)-math.log2(1e6))/4
freq_updated = starting_freq
freq_breaks = []
while freq_updated <= ending_freq:
    freq_breaks.append((3/5)*2**freq_updated)
    freq_updated += incrementing_freq

# raw waveform for merging
data_freq_range = [1*1e6, 1.5*1e6, 2*1e6, 2.75*1e6, 4*1e6]
num_freq = len(data_freq_range)
Fs = 50e6      # sampling frequency: 50 MHz

# diameter in micrometer
d = [0.35*1000, 0.4*1000, 0.45*1000, 0.5*1000, 0.55*1000, 0.6*1000]

# XY Spacing
lxy = [0.7*1000, 0.75*1000, 0.8*1000, 0.85*1000, 0.9*1000, 0.95*1000, 1.0*1000, 1.05*1000]


#choose the interested segment waveform
lower_boundary = 649    # lower boundary
upper_boundary = 2150   # upper boundary

# read and merge data with filtering
L_master = 4000     # final length after FFT
t = np.linspace(0, 80, 4000)
i = 3
j = 4
ii = 3
while i <= len(d):
    while j <= len(lxy):
        d_lxy = 'd' + str(int(d[i-1])) + '_lxy' + str(int(lxy[j-1]))
        s1 = 'Sample1'
        s2 = 'Sample2'
        s3 = 'Sample3'

        # select interested segment waveform and remove basis
        while ii <= num_freq:
            FN = 'freq' + str(int(data_freq_range[ii-1]/1000)) + 'kHz'
            file_path = '/Users/christinad/Desktop/ML/code/PnC/'
            s1_time_domain = genfromtxt(file_path+'RawData_TimeDomain.'+d_lxy+'.'+s1+'.'+FN+'.csv')
            s2_time_domain = genfromtxt(file_path + 'RawData_TimeDomain.' + d_lxy + '.' + s2 + '.' + FN + '.csv')
            s3_time_domain = genfromtxt(file_path + 'RawData_TimeDomain.' + d_lxy + '.' + s3 + '.' + FN + '.csv')
            # data check; plot time domain for paper
            plt.figure(1)
            plt.plot(t, s1_time_domain)
            plt.plot(t, s2_time_domain)
            plt.plot(t, s3_time_domain)
            plt.title('PnC Time Domain')
            plt.xlabel('Time [\musec]')
            plt.ylabel('Amplitude')
            plt.ylim([80, 170])
            plt.show()

            # plot time domain after removing basis
            plt.figure(2)
            avg1 = np.mean(s1_time_domain)
            avg2 = np.mean(s2_time_domain)
            avg3 = np.mean(s3_time_domain)
            plt.plot(t, (s1_time_domain-avg1)/100)
            plt.plot(t, (s2_time_domain - avg2))
            plt.plot(t, (s3_time_domain - avg3))
            plt.title('V_f(t)')
            plt.xlabel('time (mu sec)')
            plt.ylabel('Amplitude (mV)')
            plt.ylim([-40/100, 40/100])
            plt.show()

            # remove basis
            # sample 1
            avg_S1 = np.mean(s1_time_domain[lower_boundary:upper_boundary, 1])
            trace_data_Sample1(5).freq(ii).data(:, 1) = s1_time_domain[lower_boundary: upper_boundary, 1]-avg_S1
            # sample 2
            avg_S2 = np.mean(s2_time_domain[lower_boundary:upper_boundary, 1])
            trace_data_Sample2(5).freq(ii).data(:, 1) = s2_time_domain[lower_boundary: upper_boundary, 1]-avg_S2
            # sample 3
            avg_S3 = np.mean(s3_time_domain[lower_boundary:upper_boundary, 1])
            trace_data_Sample3(5).freq(ii).data(:, 1) = s3_time_domain[lower_boundary: upper_boundary, 1] - avg_S3

            # plot interested segment of time domain after removing basis
            plt.figure(3)
            plt.plot(t[lower_boundary: upper_boundary], trace_data_Sample1(5).freq(ii).data)
            plt.plot(t[lower_boundary: upper_boundary], trace_data_Sample2(5).freq(ii).data)
            plt.plot(t[lower_boundary: upper_boundary], trace_data_Sample3(5).freq(ii).data)
            plt.title(['All Cutted PnCs Time Domain Signal'])
            plt.title('Time Domain')
            plt.xlabel('time [micro second]')
            plt.ylabel('Amplitude')
            plt.show()
            ii += 3

        # merging with filtering
        [f1, act_cum1] = Filtering(trace_data_Sample1, Fs, L_master, num_freq, data_freq_range, freq_breaks)
        [f2, act_cum2] = Filtering(trace_data_Sample2, Fs, L_master, num_freq, data_freq_range, freq_breaks)
        [f3, act_cum3] = Filtering(trace_data_Sample3, Fs, L_master, num_freq, data_freq_range, freq_breaks)
        # merged PnC FFT
        plt.figure()
        plt.plot(f1, act_cum1)
        plt.plot(f2, act_cum2)
        plt.plot(f3, act_cum3)
        plt.plot(f1, Merged_Avg_ref_sig, 'm')
        plt.legend('Sample 1', 'Sample 2', 'Sample 3', 'Refer Sig')
        plt.title(['FFT of PnCs'])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xlim([0, 5e6])
        plt.ylim([0, 3])
        plt.show()
        plt.pause(0.4)

        # transmission spectra
        TransmissionSpectra.frequency(:, 1) = f1;
        # consider the Gain 40dB for testing PnC samples
        TransmissionSpectra.(d_lxy).(S1) = (act_cum1 / 100) / Merged_Avg_ref_sig
        TransmissionSpectra.(d_lxy).(S2) = (act_cum2 / 100) / Merged_Avg_ref_sig
        TransmissionSpectra.(d_lxy).(S3) = (act_cum3 / 100) / Merged_Avg_ref_sig

        # This is for Paper Figure DoE transmission Spectra
        plt.figure()
        plt.plot(f1, TransmissionSpectra.(d_lxy).(S1), 'b')
        plt.plot(f2, TransmissionSpectra.(d_lxy).(S2), 'k')
        plt.plot(f3, TransmissionSpectra.(d_lxy).(S3), 'r')
        plt.legend('Sample 1', 'Sample 2', 'Sample 3', 'FontName', 'Times New Roman')
        plt.title(['Transmission Spectra', ' Data: ', d_lxy])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude')
        plt.xlim([0, 2e6])
        plt.ylim([0, 0.4])
        plt.show()
        # FileName = strcat('Transmisson_Spectra_L4000_ZeroPadding', arryName);
        # saveas(gcf, FileName, 'epsc');
        plt.pause(0.4)

        j += 4
    i += 3

# functions
# Filtering freq localizing basis functions
function[freq, Final_cum] = Filtering(trace_data, Fs, L_master, num_freq, data_freq_range, freq_breaks)
starting_freq = math.log2(1e6)-4*(math.log2(4e6)-math.log2(1e6))/4
incrementing_freq = (math.log2(4e6)-math.log2(1e6))/4
ending_freq = math.log2(4e6)+(math.log2(4e6)-math.log2(1e6))/4

freq_updated = starting_freq
freq_full_range = []
while freq_updated <= ending_freq:
    freq_full_range.append(2**freq_updated)
    freq_updated += incrementing_freq

# Frequency axis
freq_temp = 0
f = []
while freq_temp <= math.floor(L_master/2):
    f.append((Fs/L_master)*freq_temp)
    freq_temp += 1

# frequency localizing basis functions(Equation 35 of Welsh 2007)
# Equation 35 of Welsh 2007
Legend = []
for q = 1:1:
    for i = 1: length(freq_full_range):\
        #frequency localizing basis functions, Equation 35 of Welsh 2007
        # in general, the F(i + 1) filter is for F(i)
        # left part of the equation
        F(i).order(q) = tf([(freq_breaks(i + 1) * 2 * pi) ** q zeros(1, q * (i))], [1])
        for k = 1:i + 1
            # coeffiecent of(s + Pi)
            den = [1 freq_breaks[k] * 2 * np.pi]
            for qq = 1:q - 1
                # conv() to get the coefficent of den 1 / (s + Pi) ^ q
                den = conv(den, [1 freq_breaks(k) * 2 * np.pi])
            F(i).order(q) = F(i).order(q) * tf([1], den)

        # frequency localizing basis function bode plot compilation
        plt.figure(1000)
        plt.bode(F(i).order(q))
        plt.show()
        [mag, phas] = bode(F(i).order(q), f * 2 * np.pi)

        plt.figure(300)
        plt.semilogx(f, np.squeeze(mag))
        # Mark peaks
        [Peak, PeakIdx] = findpeaks(np.squeeze(mag));
        # text(f(PeakIdx), Peak, sprintf('Peakf: %6.3f', f(PeakIdx)))
        plt.xlabel('f [Hz]')
        plt.title(['Frequency basis functions with q = ' + str(q)])
        Legend[i] = ('f_0 = '+str(freq_full_range[i]))
        plt.legend(Legend)
        plt.show()
# 197
# merging data with selected frequency localizing basis functions
act_cum = zeros(L_master / 2 + 1, 1)
    for i = 1:num_freq:
        L_orig = length(trace_data(5).freq(i).data(:, 1))
        act(1: L_orig) = trace_data(5).freq(i).data(:, 1)
        L_zero_padding = L_master - L_orig

        # signal and windowing
        wind(1: L_orig) = blackman(L_orig)
        act_winded = wind. * act

        # FFT
        act_winded = [act_winded, zeros(1, L_zero_padding)] # zero-padding
        L = length(act_winded)
        ACT = fft(act_winded) # window processed
        ACT2 = abs(ACT / L_orig) # normalization by interested segment signal length
        SSB = ACT2(1:floor(L / 2) + 1) # Single Side Band(SSB)
        SSB(2: end - 1) = 2 * SSB(2: end - 1)
        pro_data(5).freq(i).data(:, 1) = SSB

        # raw frequency domain data
        plt.figure(100)
        plt.subplot(5, 1, i)
        plt.plot(f, abs(pro_data(5).freq(i).data(:, 1)), 'r-', 'LineWidth', 1.5)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        Legend = ('f_{center} = '+ str(Data_freq_range(i)))
        plt.legend(Legend)
        plt.xlim([0, 5e6])
        plt.ylim([0, 8])
        plt.show()
        set(gcf, 'units', 'inches', 'position', [5, 5, 5, 10])
        # select localizing basis functions based on experimental data
        # start with  # 5 basis function (1e6 Hz)
        plt.figure(1100)
        plt.bode(F(i + 4).order(q))
        plt.show()

        [mag, phas] = bode(F(i + 4).order(q), f * 2 * np.pi)

        plt.figure(500)
        plt.semilogx(f, np.squeeze(mag))
        # Mark peaks
        [Peak, PeakIdx] = findpeaks(np.squeeze(mag))
        text(f(PeakIdx), Peak, sprintf('Peakf: %6.3f', f(PeakIdx)))
        plt.xlabel('f [Hz]')
        plt.title(['Frequency basis functions with q = ' + str(q)])
        Legend[i] = ('f_0 = ' + str(freq_full_range(i + 4)))
        plt.legend(Legend)

        # Filter data with frequency localizing basis functions
        for k = 1:length(pro_data(5).freq(i).data)
            pro_data(5).freq(i).data(k, 1) = pro_data(5).freq(i).data(k, 1) * mag(:,:, k)*exp(j * phas(:,:, k)*pi / 180);
        # frequency domain data with filtering
        plt.figure(101)
        plt.subplot(5, 1, i)
        plt.plot(f, abs(pro_data(5).freq(i).data(:, 1)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        Legend = ('f_{center} = '+str(data_freq_range[i]))
        plt.legend(Legend)
        plt.xlim([0, 5e6])
        plt.ylim([0, 4])
        # merging multiple experiments from different testing frequencies
        act_cum = act_cum + abs(pro_data(5).freq(i).data(:, 1))
# remove the first point
freq = f[2:end]
Final_cum = act_cum[2: end]

