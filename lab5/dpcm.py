from abc import ABCMeta, abstractmethod
from scipy.io import wavfile
import numpy as np
import wave


class Compress(metaclass=ABCMeta):
    def __init__(self, filename):
        self._filename = filename
        _, sig = wavfile.read('../wav_source/{}.wav'.format(filename))
        self._sig = np.array(sig, dtype=np.int16)

    @abstractmethod
    def compress(self):
        pass


class Decompress(metaclass=ABCMeta):
    def __init__(self, filename, byte):
        with open(filename, 'rb') as f:
            file = f.read()
            self._head = np.int16((file[0] << 8) + file[1])
            dif = []
            for i, x in enumerate(file):
                if i < 2:
                    continue
                if byte == 1:
                    dif.append(np.int8(x))
            self._dif = np.array(dif, dtype=np.int8)
            self._sig = self.decompress()

    @abstractmethod
    def decompress(self):
        pass

    def write_file(self, filename):
        with wave.open('decompressed1/{}.pcm'.format(filename), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(self._sig)
