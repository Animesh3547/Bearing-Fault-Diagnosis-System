# import scipy.io as sio
# import numpy as np
# import matplotlib.pyplot as plt

# # load file
# normal_0hp = sio.loadmat('../data/normal/normal_0hp.mat')
# inner_race_0hp = sio.loadmat('../data/inner/ir_007_0hp.mat')
# outer_race_0hp = sio.loadmat('../data/outer/or_007_0hp.mat')
# ball_0hp = sio.loadmat('../data/ball/ball_007_0hp.mat')

# # print keys
# print(normal_0hp.keys())
# print(inner_race_0hp.keys())
# print(outer_race_0hp.keys())
# print(ball_0hp.keys())

# # extract vibration signal

# fixed_length = 120000   # 10 sec at 12kHz


# signal_0 = normal_0hp['X097_DE_time'].squeeze()
# signal_0 = signal_0[:fixed_length]
# signal_1 = inner_race_0hp['X105_DE_time'].squeeze()
# signal_1 = signal_1[:fixed_length]
# signal_3 = outer_race_0hp['X130_DE_time'].squeeze()
# signal_3 = signal_3[:fixed_length]
# signal_2 = ball_0hp['X118_DE_time'].squeeze()
# signal_2 = signal_2[:fixed_length]

# print(signal_0.shape)
# print(signal_1.shape)
# print(signal_3.shape)
# print(signal_2.shape)


# # plot raw signal
# # plt.plot(signal_0[:2000])
# # plt.title("Raw Vibration Signal of Normal Bearing (0hp)")
# # plt.show()
# # plt.plot(signal_1[:2000])
# # plt.title("Raw Vibration Signal of Inner Race Fault (0hp)")
# # plt.show()
# # plt.plot(signal_2[:2000])
# # plt.title("Raw Vibration Signal of Ball Fault (0hp)")
# # plt.show()
# # plt.plot(signal_3[:2000])
# # plt.title("Raw Vibration Signal of Outer Race Fault (0hp)")
# # plt.show()

# from scipy.signal import spectrogram

# fs = 12000  # sampling frequency
# nperseg = 256
# noverlap = 128

# frequencies, times, Sxx = spectrogram(signal_0, fs, nperseg=nperseg, noverlap=noverlap)

# Sxx = np.log(Sxx + 1e-10)

# plt.pcolormesh(times, frequencies, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title("Spectrogram of Normal Bearing (0hp)")
# plt.show()

# frequencies, times, Sxx = spectrogram(signal_1, fs, nperseg=nperseg, noverlap=noverlap)
# Sxx = np.log(Sxx + 1e-10)

# plt.pcolormesh(times, frequencies, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title("Spectrogram of Inner Race Fault (0hp)")
# plt.show()

# frequencies, times, Sxx = spectrogram(signal_2, fs, nperseg=nperseg, noverlap=noverlap)
# Sxx = np.log(Sxx + 1e-10)

# plt.pcolormesh(times, frequencies, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title("Spectrogram of Ball Fault (0hp)")
# plt.show()

# frequencies, times, Sxx = spectrogram(signal_3, fs, nperseg=nperseg, noverlap=noverlap)
# Sxx = np.log(Sxx + 1e-10)

# plt.pcolormesh(times, frequencies, Sxx)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.title("Spectrogram of Outer Race Fault (0hp)")
# plt.show()