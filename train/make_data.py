from skimage import io
import os
from torch.utils.data import Dataset


def make_label(root_name):
    with open('../data/{}_label.txt'.format(root_name), 'w') as f:
        for root, dirs, files in os.walk('../data/' + root_name):
            for file in files:
                if file[-3:] != 'bmp':
                    continue
                f.write(os.path.join(root, file)+'!'+str(root.split('\\')[1])+'\n')


class MyDataset(Dataset):
    def __init__(self, names_file, transform=None):
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.names_list[idx].split('!')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path)

        label = int(self.names_list[idx].split('!')[1])

        sample = {'image': image, 'label': label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


if __name__ == '__main__':
    make_label('training-set')
    make_label('test-set')
