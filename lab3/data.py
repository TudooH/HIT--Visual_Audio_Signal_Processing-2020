import urllib.request as ur
import pandas as pd
from PIL import Image

data = pd.read_json('../img/data.json', lines=True)
del data['extras']
data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)
del data['annotation']

with open('../data/points.txt', 'w') as f:
    for line in data['points']:
        f.write('{} {} {} {}\n'.format(line[0]['x'], line[0]['y'], line[1]['x'], line[1]['y']))

for index, line in data.iterrows():
    print(index)
    try:
        img = ur.urlopen(line[0])
        img = Image.open(img)
        img = img.convert('RGB')
        img.save('../data/{}.jpeg'.format(index), 'JPEG')
    except:
        print('{} error'.format(index))
