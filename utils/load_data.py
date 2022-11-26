from torch.utils.data import Dataset
import numpy as np

class Load_Transformed_Dataset(Dataset):
    def __init__(self, training_flist, transform=None):
        self.training_flist = training_flist
        self.transform = transform

    def __len__(self):
        return len(self.training_flist)

    def __getitem__(self, idx):
        img_dir = self.training_flist[idx]
        img_arr = np.load(img_dir) #channel, h, w
        #not sure why, but img_arr need to change to h,w,c to make the transform work
        img_arr = np.moveaxis(img_arr,0,-1)
        #print(img_arr.shape)
        if self.transform:
            img_arr = self.transform(img_arr)
            #img_arr = img_arr[0] #get the tensor from list
            
        return img_arr