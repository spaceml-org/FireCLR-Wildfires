import numpy as np
from torchvision import transforms

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
    
class ContrastiveLearningDataset:
    def __init__(self, num_trans_func,list_trans):
        self.num_trans_func = num_trans_func
        self.list_trans = list_trans

    def random_number(self):
        return np.random.randint(0,self.num_trans_func)

    def get_trans_func(self):
                    
        idx1 = self.random_number()
        idx2 = self.random_number()
        
        img_transforms = transforms.Compose([transforms.ToTensor(), #must have to read the array as a PIL image, then could be used as an input to transforms
                                        self.list_trans[idx1],
                                        self.list_trans[idx2]])
        
        return img_transforms 