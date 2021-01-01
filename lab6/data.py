import os

from lab6.detection import Detection


# de-noise with vad
for i in range(5):
    det = Detection('{}'.format(i+1))
    det.detect()
    for j in range(9):
        det = Detection('{}_{}'.format(i+1, j+1))
        det.detect()

# modify list.scp
with open('mfcc/list.scp', 'w') as f:
    for i in range(5):
        f.write('../wav/{}.wav ../mfc/{}.mfc\n'.format(i + 1, i + 1))
        for j in range(9):
            f.write('../wav/{}_{}.wav ../mfc/{}_{}.mfc\n'.format(i + 1, j + 1, i + 1, j + 1))

# generate mfc files
os.system('cd mfcc && hcopy -A -D -T 1 -C tr_wav.cfg -S list.scp')
