import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class TestDataset_Batch(data.Dataset): 
    def __init__(self, image_root, trainsize):
        self.trainsize = trainsize
        image_root_dir_list = sorted(os.listdir(image_root))
        self.images = [image_root + f for f in image_root_dir_list if f.endswith('.jpg')]
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        size = image.size
        image = self.img_transform(image)
        name = self.images[index].split('/')[-1].replace('.jpg','.png')

        return image, name, size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size

def get_loader_testbatch(image_root, batchsize, trainsize, shuffle=False, num_workers=8, pin_memory=True):
    dataset = TestDataset_Batch(image_root,  trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  drop_last=False)
    return data_loader
