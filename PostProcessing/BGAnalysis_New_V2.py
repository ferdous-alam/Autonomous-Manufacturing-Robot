import control as control
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import fft

plt.close('all')


# This code is used for FFT, Data collection
A = pd.read_csv('DOE_avg_reference_signal_timedomain.csv')  # Reference signal
B = pd.read_csv('DOE_Ref_SampleA_300V_0dB_50avg_freq-2250kHz_location-7_Block-1_Jan-19-2020_11-18.csv')  # actual signal

# signal range in time domain. This is lower and upper values are determined by ploting the time domain data to
# determine where the pulse starts and ends
SigRangeA_lower = 550
SigRangeA_upper = 1050

SigRangeB_lower = 800
SigRangeB_upper = 2040

Sampling_frequency = 50000000  # Sampling Frequency (MHz): 50; 25; 12.5; 6.25

n = len(A)+1       # number of data points !for some reason len function is returning 3999 instead of 4000
Total_time = n/Sampling_frequency  # total time
dt = Total_time/n               # difference in time for plotting

# time for measurements of A and B csv files, goes to n-1 bc at n it repeats !check this
index = 0
t = [None]*(n-1)
while index < n:
    t.append(dt*index*10**6)
    index += 1

# calculating FFT

df = 1/dt  # difference between frequency

# set mean value before and after pulse, eliminates noise and reflections for A and B
No_Reflections_A = [np.mean(A)]*n  # array of mean A
No_Reflections_A[SigRangeA_lower:SigRangeA_upper] = A[SigRangeA_lower:SigRangeA_upper]  # adds the pulse

No_Reflections_B = [np.mean(B)]*n  # array of mean B
No_Reflections_B[SigRangeB_lower:SigRangeB_upper] = B[SigRangeB_lower:SigRangeB_upper]  # adds the pulse


# take fft A
fft_A = abs(fft(No_Reflections_A))
index = int(n/2)
while len(fft_A) != (n/2):
    fft_A = np.delete(fft_A, index)

# create frequency depth points for plotting
frq_A_points = [None] * int(n/2)
index = 0
while index < (int(n/2)):
    frq_A_points[index] = index * df/n * 10**-6
    index += 1

# normalize coefficents of fft of A
index = 0
fft_A_normalized = [None]*len(fft_A)
fft_A_max = max(fft_A[2:n])
while index < int(n/2):
    fft_A_normalized[index] = fft_A[index]/fft_A_max
    index += 1

# ignoring noise for fft A
index = 1
while index < len(fft_A_normalized):
    if fft_A_normalized[index] >= .06:
        fANindex_B = index  # idk what to rename this
        break
    index += 1
index = len(fft_A_normalized)-1
while index > 1:
    if fft_A_normalized[index] >=.06:s
        fANindex_E = index
        break
    index -= 1


# take fft B
fft_B = abs(fft(No_Reflections_B))
index = int(n/2)
while len(fft_B) != (n/2):
    fft_B = np.delete(fft_B, index)

# create frequency depth points for plotting
frq_B_points = [None] * int(n/2)
index = 0
while index < (int(n/2)):
    frq_B_points[index] = index * df/n * 10**-6
    index += 1

# normalize coefficents of fft of B
index = 0
fft_B_normalized = [None]*len(fft_B)
fft_B_max = max(fft_B[2:n])
while index < int(n/2):
    fft_B_normalized[index] = fft_B[index]/fft_B_max
    index += 1

#  plotting fft ISNT WORKING
plt.plot(frq_A_points, abs(fft_A))
plt.plot(frq_B_points, abs(fft_B))
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [au]')
plt.title('Frequency Domain')
plt.xlim([0.2, 5.0])
plt.legend(['Reference Signal','Sample (d=0.00mm, a=0.00mm)'])
plt.grid()
plt.show()

# 121 Converts amplitude ratio to decibels !!!!clean this up!!!!
dbA = control.mag2db(fft_A_normalized)
mA = len(dbA)
dbB = control.mag2db(fft_B_normalized)
mB = len(dbB)

dbNA = dbA[1:mA]
Mx = max(dbNA)
My = Mx - 6
index = 0
while index < mA:
    if dbA[index] == Mx:
        index_of_max = index
    index += 1
Rmax = index_of_max
index = 0

while index < mA:
    if dbA[index] < My and index < index_of_max:
        N1 = index
    if dbA[index] > My and index > index_of_max:
        N2 = index
    index += 1

L1A = frq_A_points[N1] * 10**-6
L2A = frq_A_points[N2] * 10**-6
CfA = (L1A+L2A)/2
PkA = frq_A_points[Rmax]*10**-6
dbNB = dbB[1:mB]
Mx = max(dbNB)
My = Mx - 6

index = 0
while index < mB:
    if dbB[index] == Mx:
        index_of_max = index
    index += 1

index = 0
while index < mB:
    if dbB[index] <= My and index < index_of_max:
        N1 = index
    if dbB[index] >= My and index >index_of_max:
        N2 = index
        if dbB[index + 1] < My:
            break
L1B = frq_B_points[N1]*10**-6
L2B = frq_B_points[N2]*10**-6
CfB = (L1B+L2B)/2
PkB = frq_A_points[index_of_max]*10**-6

#  calculating transfer function
FAf = frq_A_points[1:mA-1]
num = fft_A[1:mA-1]
den = fft_B[1:mB]/100  # denominator of transfer function
TNF = den/num  # output over input
dbTNF = control.mag2db(TNF) #transfer function converted to decibels
k1 = 1  # counts number of frequencies contributions below 0.5 MHz
k2 = 1  # counts number of frequencies contributions above 3.5 MHz

while FAf(k1) < 500000: #Lower Limit 0.5 MHz
    k1= k1 + 1
while FAf(k2) <= 3500000:  #Upper Limit 3.5 MHz
    k2 = k2 + 1

#  Calculating Peak & Central Frequency of the Transfer Function Plots
dbTNF_lim = dbTNF[k1:k2]   # limit range of transfer function
FAf_lim = FAf[k1:k2]       # corresponding frequencies for the range
mC = len(dbTNF_lim)    # length of the limit
Mx = max[dbTNF_lim]        # max of transfer function
My = Mx - 6                # DB threshold

index = 0
while index<mC:
    if dbTNF_lim(index) == Mx:
        index_of_max = index
    index += 1
index=0
while index<mC:
    if dbTNF_lim(index) < My and index < index_of_max:
        N1 = index
if dbTNF_lim(index) > My and index > index_of_max:
    N2 = index

L1TF = FAf_lim[N1]*10**-6  # converts MHz to Hz
L2TF = FAf_lim[N2]*10**-6  # converts MHz to Hz
CfTF = (L1TF+L2TF)/2     # average
PkTF = FAf_lim[index_of_max]*10**-6  # converts MHz to Hz

# Plotting Transfer Function PLots
plt.plot(FAf*10**-6, dbTNF)
plt.xlim([0.5, 4.5])
plt.ylim([-70, 20])
plt.xlabel('Frequency [MHz]')
plt.ylabel('Amplitude [dB]')
plt.title('Transfer Func (Decibel Scale)')
plt.grid()
plt.show()

# store transfer function data