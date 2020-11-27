import os

from lab3.search import search


def iou(a, b):
    x = min(a[2], b[2]) - max(a[0], b[0])
    y = min(a[3], b[3]) - max(a[1], b[1])
    if x <= 0 or y <= 0:
        return 0

    area = x * y
    area_a = (a[3] - a[1]) * (a[2] - a[0])
    area_b = (b[3] - b[1]) * (b[2] - b[0])
    return area / (area_a + area_b - area)


points = []
for line in open('../data/points.txt', 'r'):
    lines = line.split(' ')
    points.append([float(lines[0]), float(lines[1]), float(lines[2]), float(lines[3][: -1])])

tot = 0
test = []
for root, dirs, files in os.walk('../data'):
    for file in files:
        if file.endswith('.jpeg'):
            x1, y1, x2, y2, _, _ = search(os.path.join(root, file))
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                tot += 1
            test.append([x1, y1, x2, y2])

ac = 0
with open('result.txt', 'w') as f:
    for i in range(len(points)):
        rate = iou(points[i], test[i])
        f.write('{} {} {}\n'.format(i, rate, test[i]))
        if rate > 0.1:
            ac += 1

print(ac)
print(ac / len(points))
