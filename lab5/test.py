from scipy.io import wavfile
import numpy as np

from lab5.dpcm1 import Compress1, Decompress1
from lab5.dpcm2 import Compress2, Decompress2


def cal_snr(ori, com):
    if len(ori) != len(com):
        ori = ori[: -1]
    ori = np.array(ori, dtype=np.int64)
    com = np.array(com, dtype=np.int64)
    print(ori)
    print(com)
    dif = ori - com

    return 10 * np.log10(sum(np.power(ori, 2)) / sum(np.power(dif, 2)))


# for i in range(10):
#     co = Compress1(str(i+1), 100)
#     co.compress()
#     de = Decompress1('compressed1/{}'.format(str(i+1)), 100)
#
# for i in range(10):
#     co = Compress2(str(i+1))
#     co.compress()
#     de = Decompress2('compressed2/{}'.format(str(i+1)))

for i in range(10):
    if i != 2:
        continue
    _, sig = wavfile.read('../wav_source/{}.wav'.format(str(i+1)))
    _, compressed1 = wavfile.read('decompressed1/{}.pcm'.format(str(i+1)))
    _, compressed2 = wavfile.read('decompressed2/{}.pcm'.format(str(i+1)))
    # print(i+1, cal_snr(sig, compressed1), cal_snr(sig, compressed2))
