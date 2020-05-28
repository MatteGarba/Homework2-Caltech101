from torchvision.datasets import VisionDataset
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        
        self.mapping = dict()                   # couples class_label : Progressive_ID
        self.id = 0                             # the progressive ID to be associated to the classes
        self.discard = "BACKGROUND_Google"      # the class to be disscarded
        self.data = list()                      # list of tuples: (img, id)
        
        if split != 'train' and split != 'test':
            print("WARNING: invalid split name provided! (Using the default one: TRAIN)")
            split = "train"
        
        pathToAccess = "./Caltech101/"+split+".txt"
        with open(pathToAccess, "r") as f:
            for line in f.readlines():
                
                class_label = (line.split("/"))[0]          # add the class to the mapping if it's missing
                if class_label != self.discard:             # remove the background
                    if class_label not in self.mapping:
                        self.mapping[class_label] = self.id
                        self.id = self.id + 1

                    ID = self.mapping[class_label]              # extract the id of the class
                    img = pil_loader(root+"/"+line[:-1])        # load the image from the dataset
                    tupl = (img, ID)
                    self.data.append(tupl)                      # append the couple to the dataset
                
        
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.data[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
    
    def stratified_sampling(self, train_size=0.5):
        ind = [i for i in range(len(self.data))]    # list of indexes of the tuples (img, id) where id is the numeric corresp of the class label
        IDs = list()                                # extract the list of ids in order to stratify over it (maintain the proportions)
        for t in self.data:
            IDs.append(t[1])
        stratified_train_ind, stratified_val_ind = train_test_split(ind, train_size=train_size, stratify=IDs) # stratified sampling
        return stratified_train_ind, stratified_val_ind
