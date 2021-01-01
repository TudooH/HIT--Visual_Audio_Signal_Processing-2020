from lab6.dtw import get_feature, dtw


# get 5 form features
features = []
for i in range(5):
    features.append(get_feature(str(i+1)))

# test other 45 features, write the result at the result.txt
correct = 0
with open('result.txt', 'w') as f:
    for i in range(5):
        for j in range(9):
            label = -1
            dis = 0xffff
            for k, feature in enumerate(features):
                tmp = dtw(feature, get_feature('{}_{}'.format(i+1, j+1)))
                if tmp < dis:
                    label = k
                    dis = tmp
            f.write('{}_{}: {}\n'.format(i+1, j+1, label+1))
            if i == label:
                correct += 1
    f.write('\ncorrected: {}, total: {}\n'.format(correct, 45))
    f.write('final acc rate is %.2f%%\n' % (correct * 100 / 45.))

print('acc: %.2f%%' % (correct * 100 / 45.))
