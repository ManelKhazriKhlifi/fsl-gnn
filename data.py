import torch
import os
import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ImageFolderLoader(Dataset):
    def __init__(self, root, partition='train', device='cuda'):
        super(ImageFolderLoader, self).__init__()
        self.root = root
        self.partition = partition
        self.device = device
        
        self.data = self.load_dataset()

    def load_dataset(self):
        dataset_path = os.path.join(self.root, self.partition)
        dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform())
        return dataset
    
    def transform(self):
        # Set normalizer
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                         std=[0.5, 0.5, 0.5])
        # Set transformer
        if self.partition == 'train':
            return transforms.Compose([transforms.RandomResizedCrop(84),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(45), 
                                    #    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    #    transforms.RandomGrayscale(p=0.1),
                                    #    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                                    #    transforms.RandomErasing(),
                                       transforms.ToTensor(),
                                       normalize])
        else:  # 'val' or 'test'
            return transforms.Compose([transforms.Resize(90),
                                       transforms.CenterCrop(84),
                                       transforms.ToTensor(),
                                       normalize])

        
    def get_batch(self, num_tasks, num_ways, num_shots, num_queries, seed=None):
        if seed is not None:
            random.seed(seed)

        support_data, support_label, query_data, query_label = [], [], [], []

        for _ in range(num_ways * num_shots):
            support_data.append([])
            support_label.append([])
        for _ in range(num_ways * num_queries):
            query_data.append([])
            query_label.append([])

        full_class_list = list(range(len(self.data.classes)))

        for t_idx in range(num_tasks):
            task_class_list = random.sample(full_class_list, num_ways)

            for c_idx in range(num_ways):
                class_data_list = []
                class_idx = task_class_list[c_idx]
                for sample_idx in range(num_shots + num_queries):
                    sample = random.choice(self.data.targets)
                    while sample != class_idx:
                        sample = random.choice(self.data.targets)
                    img, _ = self.data[sample]
                    class_data_list.append(img)

                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots].append(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots].append(c_idx)

                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries].append(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries].append(c_idx)
                    

        support_data = torch.stack([torch.stack(data).to(self.device) for data in support_data], 1)
        support_label = torch.stack([torch.tensor(label).to(self.device) for label in support_label], 1)
        query_data = torch.stack([torch.stack(data).to(self.device) for data in query_data], 1)
        query_label = torch.stack([torch.tensor(label).to(self.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]
    
def get_loaders(root, device):
    return {'train': ImageFolderLoader(root=root, partition='train', device=device), 
            'val': ImageFolderLoader(root=root, partition='val', device=device), 
            'test': ImageFolderLoader(root=root, partition='test', device=device)}