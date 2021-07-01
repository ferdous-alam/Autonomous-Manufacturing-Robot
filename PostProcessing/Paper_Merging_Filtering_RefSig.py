import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import math
from numpy import genfromtxt
from scipy import signal

from scipy.signal.ltisys import TransferFunction as TransFun
from numpy import polymul,polyadd
# functions

# Filtering freq localizing basis functions
def Filtering(trace_data, Fs, num_freq, Data_freq_range, freq_breaks, L_master):
    starting_freq = math.log2(1e6) - 4 * (math.log2(4e6) - math.log2(1e6)) / 4
    incrementing_freq = (math.log2(4e6) - math.log2(1e6)) / 4
    ending_freq = math.log2(4e6) + (math.log2(4e6) - math.log2(1e6)) / 4
    freq_full_range = []
    freq_updated = starting_freq
    while freq_updated <= ending_freq:
        freq_full_range.append(2 ** freq_updated)
        freq_updated += incrementing_freq
    # Frequency axis
    freq_temp = 0
    f = []
    while freq_temp <= math.floor(L_master / 2):
        f.append((Fs / L_master) * freq_temp)
        freq_temp += 1
    i = 0
    qq = 1
    F = []
    while i < len(freq_full_range):
        # F(i + 1) filter is for F(i)
        # left part of the equation
        F.append(TransFun([(freq_breaks[i] * 2 * np.pi)**1, np.zeros(1, dtype=float,)], 1))
        k = 0
        while k < i+1:
            den = [1, freq_breaks[k]*2*np.pi]   # coefficient of (s+Pi)
            while qq >= 0:
                # conv() to get the coefficient of den 1/(s+Pi)^q
                den = np.convolve(den, [1, freq_breaks[k]*2*np.pi])
                qq -= 1

            # this line isn't working correctly ive also tried matmul
            F[i] = np.dot((F[i]).to_ss(), (TransFun([1], den)).to_ss())
            k += 1
        bode_f = 2 * np.pi * np.asarray(f)

        # this line isnt working error message: "input must be a rank-1 array" but they are both rank 1
        [mag, phas] = signal.bode(F[i], bode_f)

        i += 1
    # merging data with selected frequency localizing basis functions
    act_cum = np.zeros(int(L_master/2 + 1))
    i = 1
    act = []
    wind = []

    while i <= num_freq:
        L_orig= len(trace_data)
        act[0:L_orig-1] = trace_data
        L_zero_padding = L_master- L_orig

        # signal and windowing
        wind[0:L_orig-1] =np.blackman(L_orig)
        j=0
        act_winded = []
        while j<len(act):
            act_winded.append(wind[j]*act[j])
            j += 1
        # FFT
        #act_winded=[act_winded, np.zeros(L_zero_padding)]   # zero-padding

        L = len(act_winded)
        ACT = np.fft.fft(act_winded)   #window processed
        ACT2 = abs(ACT/L_orig)  # normalization by interested signal length
        SSB = ACT2[0: math.floor(L/2)]
        SSB[1:len(SSB)-1] = 2*SSB[1:len(SSB)-1]
        pro_data=SSB
        # [mag, phas] = signal.bode(F[i+3], f*2*np.pi)

        k = 0
        while k < len(pro_data):
            pro_data[k] = pro_data[k]*mag[k]*math.exp(j*phas[k]*np.pi/180)
            k += 1
        i += 1
        #merging multiple experiments from different testing frequencies
        act_cum = act_cum + abs(pro_data)
    return act_cum


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
freq_full_range = []
freq_updated = starting_freq
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

# choose the interested segment waveform
lower_boundary = 540    # lower boundary
upper_boundary = 1101   # upper boundary
L_master = 4000         # final length after FFT
# Frequency axis
freq_temp = 0
f = []
while freq_temp <= math.floor(L_master/2):
    f.append((Fs/L_master)*freq_temp)
    freq_temp += 1
sig_cum = np.zeros(int(L_master/2+1))

# read and merge with data with filtering
t = np.linspace(0, 80, 4000)
TableN = 12     # the total number of table run is 12
jj = 1
ii = 1
colors1 = ['aqua', 'aquamarine', 'blue', 'blueviolet', 'cadetblue', 'chartreuse', 'cornflowerblue', 'darkblue', 'darkcyan', 'darkgreen', 'deepskyblue', 'black']
colors2 = ['magenta', 'chocolate', 'coral', 'crimson', 'darkmagenta', 'darksalmon', 'deeppink', 'goldenrod', 'hotpink', 'indianred', 'brown', 'mediumvioletred']
while jj <= TableN:
    # select interested segment waveform and remove basis
    while ii < num_freq:
        FN = 'freq' + str(int(data_freq_range[ii]/1000)) +'kHz'
        TN = 'Test'+ str(jj)
        file_path = '/Users/christinad/Desktop/ML/code/Ref_Sig/'
        dataset_A = genfromtxt(file_path+'RawData_TimeDomain.SampleA.' + TN + '.' + FN +'.csv')
        dataset_B = genfromtxt(file_path+'RawData_TimeDomain.SampleB.' + TN + '.' + FN +'.csv')
        # data check; plot time domain
        plt.figure(1)
        plt.plot(t, dataset_A, color=colors1[jj-1])
        plt.plot(t, dataset_B, color=colors2[jj-1])
        plt.title('Time Domain')
        plt.xlabel('time [micro second]')
        plt.ylabel('Amplitude')
        plt.savefig('/Users/christinad/Desktop/ML/code/Figures/ref_sig/time_domain.png')
        # Plot time domain after removing basis for paper
        plt.figure(2) # jj = 2; ii=3
        avgA = np.mean(dataset_A)
        avgB = np.mean(dataset_B)
        plt.plot(t, dataset_A - avgA)
        plt.title('V_r(t)')
        plt.xlabel('time (mu sec)')
        plt.ylabel('Amplitude (mV)')
        plt.ylim([-40, 40])
        plt.savefig('/Users/christinad/Desktop/ML/code/Figures/ref_sig/time_domain_removed_basis.png')
        # remove basis
        # sample A
        avgA = np.mean(dataset_A[lower_boundary: upper_boundary])
        trace_dataA = dataset_A[lower_boundary:upper_boundary]-avgA
        # sample B
        avgB = np.mean(dataset_B[lower_boundary: upper_boundary])
        trace_dataB = dataset_B[lower_boundary:upper_boundary]-avgB
        ii += 1
    # merging with filtering
    act_cumA = Filtering(trace_dataA, Fs, num_freq, data_freq_range, freq_breaks, L_master)
    act_cumB = Filtering(trace_dataB, Fs, num_freq, data_freq_range, freq_breaks, L_master)
    sig_cum = sig_cum + .5*(act_cumA+act_cumB)

    plt.figure(3)
    plt.plot(f, act_cumA, color='b')
    plt.plot(f, act_cumB, color='r')
    plt.title(['Merged reference signals'])
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Amplitude')
    plt.xlim([0, 5e6])
    plt.ylim([0, 0.5])
    plt.savefig('/Users/christinad/Desktop/ML/code/Figures/ref_sig/merged_ref_signals.png')
    ii = 1
    jj += 1
plt.show()
plt.show()
plt.show()

# merged average reference signal
avg_ref_sig = 1/TableN*sig_cum
# remove the first point
freq = f[1:]
Merged_Avg_ref_sig = avg_ref_sig[1:]

plt.figure(5)
plt.plot(freq, Merged_Avg_ref_sig, color='k')
plt.title('Averaged Reference Signal with Merged Multiple Datasets')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude')
plt.xlim([0, 5e6])
plt.savefig('/Users/christinad/Desktop/ML/code/Figures/ref_sig/averaged_signal.png')
plt.show()


