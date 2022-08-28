from sklearn.model_selection import train_test_split
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from config import get_config
from torch.utils.data import random_split
import torch

config = get_config()

class DatasetLoader:
    
    def __init__(self, ds=None, valid=False):
        
        if ds == 'cifar10':
            self.dataset = datasets.CIFAR10
        elif ds == 'cifar100':
            self.dataset = datasets.CIFAR100
        elif ds == 'fashion_mnist':
            self.dataset = datasets.FashionMNIST
        else:
            self.dataset = ds

        self._name = ds
        
        # CIFAR-10
        self.transform = {
            'train' : 
            # transforms.Compose(
            #     [transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), 
            transforms.Compose([
                      transforms.Pad(4),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomCrop(32),
                      transforms.ToTensor(),
                    #   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                      ] # Normalization for CIFAR 10
            ),
            'test'  : transforms.ToTensor()
            
            # transforms.Compose([
            #         #   transforms.Pad(4),
            #         #   transforms.RandomHorizontalFlip(),
            #         #   transforms.RandomCrop(32),
            #           transforms.ToTensor(),
            #           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Normalization for CIFAR 10
            #         #   ,transforms.Normalize((0.5,), (0.5,)) # Normalization for Fashion MNIST
            #           ]  
            # )
        } 

        # Fashion MNIST Transforms
        # if ds == 'fashion_mnist':
        #     self.transform  = {
        #         'train': transforms.Compose([
        #             #transforms.Resize(28),
        #             transforms.RandomHorizontalFlip(),
        #             #transforms.Grayscale(3), 
        #             transforms.ToTensor(), 
        #             #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #         ]),
        #         'test': transforms.Compose([
        #             #transforms.Resize(28),
        #             #transforms.Grayscale(3),
        #             transforms.ToTensor(), 
        #             #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #         ])
        #     }   

    def _prepare_dataset(self, valid=False):

        training_data = self.dataset(
            root=config.dataset_path,
            train=True,
            download=True,
            transform=self.transform['train']
        )

        if valid:
            train_indices, val_indices = train_test_split(list(range(len(training_data.targets))), test_size=0.2, stratify=training_data.targets, random_state=42)
            train_dataset = torch.utils.data.Subset(training_data, train_indices)
            valid_dataset = torch.utils.data.Subset(training_data, val_indices)

        test_dataset = self.dataset(
            root=config.dataset_path,
            train=False,
            download=True,
            transform=self.transform['test']
        )

        print("Test: ", str(len(test_dataset)))
        if valid:
            print("Train: ", str(len(train_dataset)))
            print("Valid: ", str(len(valid_dataset)))
            return train_dataset, valid_dataset, test_dataset
        else:
            print("Train: ", str(len(training_data)))
            return training_data, test_dataset
    
    def getDataLoader(self, valid=False):
        
        data = self._prepare_dataset(valid)

        train_data = data[0]
        train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

        if valid:
            valid_data = data[1]
            test_data = data[2]
            valid_dataloader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
        else:
            test_data = data[1]
            test_dataloader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)


        if valid:
            return train_dataloader, valid_dataloader, test_dataloader
        else:
            return train_dataloader, test_dataloader