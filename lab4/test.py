import numpy as np
import scipy.signal as signal


framerate = 44100
time = 10
t = np.arange(0, time, 1.0/framerate)
wave_data = signal.chirp(t, 100, time, 1000, method='linear') * 10000
wave_data = wave_data.astype(np.short)
print(wave_data)
