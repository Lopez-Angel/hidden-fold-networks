# Copyright 2021 Angel Lopez Garcia-Arias

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        https://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import numpy as np


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        

        num_samples = len(train_dataset)
        valid_samples = int(np.floor(args.split_valid * num_samples))
        train_samples = num_samples - valid_samples
        
        indices = np.arange(0,num_samples) 
        np.random.shuffle(indices)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, #shuffle=True, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[:train_samples]), **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, #shuffle=True, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[-valid_samples:]), **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )


class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        
        num_samples = len(train_dataset)
        valid_samples = int(np.floor(args.split_valid * num_samples))
        train_samples = num_samples - valid_samples
        
        indices = np.arange(0,num_samples) 
        np.random.shuffle(indices)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, #shuffle=True, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[:train_samples]), **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, #shuffle=True, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[train_samples:]), **kwargs
        )

        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )