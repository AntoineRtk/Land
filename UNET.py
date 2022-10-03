# -*- coding: utf-8 -*-

"""
@author Antoine Ratouchniak
"""

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

import Blur

import random

class Block(nn.Module):
    
    def __init__(self, size_input, size_output):
        super().__init__()
        # A block perform a convolution
        self.c1 = nn.Conv2d(in_channels = size_input,
                            out_channels = size_output,
                            kernel_size = (3, 3),
                            stride = 1,
                            padding = 1,
                            padding_mode = 'circular', # We consider images as periodic
                            bias = False)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            # We use Xavier Normal initialization
            nn.init.xavier_normal_(self.c1.weight.data)
    
    def forward(self, x):
        x = self.c1(x)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        # The encoder down samples the image to a certain size
        self.encoder = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.old = []
        self.ch_size = len(channels)
        block = Block(1, channels[0])
        self.encoder.append(block)
    
    def forward(self, x):
        for i in range(self.ch_size):
            x = self.encoder[0](x)
            self.old.append(x) # We keep the image obtained in the contracting path in storage
            x = self.pool(x)
        return (x, self.old)

class Decoder(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        # The decoder up samples the image to a certain size
        self.up_conv = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.ch_size = len(channels)
        for i in range(len(channels)):
            out_c = channels[i] if (i == len(channels) - 1) else channels[i + 1]
            up = nn.ConvTranspose2d(in_channels = channels[i],
                                    out_channels = out_c,
                                    kernel_size = (2, 2),
                                    stride = (2, 2),
                                    bias = False)
            block = Block(channels[i] * 2, out_c)
            self.up_conv.append(up)
            self.decoder.append(block)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            for conv in self.up_conv:
                nn.init.xavier_normal_(conv.weight.data)
        
    def forward(self, x, old):
        i = 0
        for up, block in zip(self.up_conv, self.decoder):
            x = up(x) # Up-sampling
            od = old[-1 - i] # We retrieve the image from the contracting path
            x = torch.cat((x, od), dim = 1) # We concatenate the two images
            x = block(x)
            i = i + 1
        return x

class UNET(nn.Module):
    
    def __init__(self, channels):
        super(UNET, self).__init__()
        
        self.encoder = Encoder(channels)
        
        self.bottleneck = Block(channels[-1], channels[-1])
        
        self.decoder = Decoder(channels[::-1])
        
        self.end = nn.Conv2d(in_channels = channels[0],
                            out_channels = 1,
                            kernel_size = 1,
                            bias = False)
        
        pytorch_total_params = sum(p.numel() for p in self.encoder.parameters())
        print('parameters encoder u_net :',pytorch_total_params)
        pytorch_total_params = sum(p.numel() for p in self.bottleneck.parameters())
        print('parameters bottleneck u_net :',pytorch_total_params)
        pytorch_total_params = sum(p.numel() for p in self.decoder.parameters())
        print('parameters decoder u_net :',pytorch_total_params)
        
    def forward(self, x):
        # Encoder part
        x, old = self.encoder(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder part
        x = self.decoder(x, old)
        
        # Final convolution
        x = self.end(x)
        
        return x

channels = [1] * 9

u_net = UNET(channels)

pytorch_total_params = sum(p.numel() for p in u_net.parameters())
print('total parameters u_net :',pytorch_total_params)
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img, label, name):
        self.img = img
        self.label = label
        self.name = name
        
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):
        img = self.img[index].float()
        label = self.label[index].float()
        if(img.size()[0] != 512):
            img = self.transform(img)
            label = self.transform(label)
        return img, label.float()
    
    transform = transforms.Compose([
        transforms.Resize((512, 512))
        ])
    
def load(sgm, seed = random.randint(0, 1000)):
    img, label, names = [], [], []
    for i in range(14):
        path = 'train/train'+str(i)+'.npy'
        img.append(torch.from_numpy(np.load(path)[None, :]))
        label.append(torch.from_numpy(Blur.blur_gaussian(np.load(path),
                                                sgm)[None, :]))
        names.append(str(i))
        random.seed(seed)
    return Dataset(img, label, names), seed

def train(u_net):
    train = False # Train the U-Net
    difference = False # Show the difference between the result and the label
    save = True # Save the parameters of the U-Net
    valid = False # Show the valid loss
    sgm = 5 # Standard deviation of the Gaussian
    ls = []
    seed = 0
    # Load a save
    # checkpoint = torch.load('save/save_6000.pth')
    # u_net.load_state_dict(checkpoint['model'])
    # ls = checkpoint['loss']
    # seed = checkpoint['seed']
    if(len(ls) > 0 and not train):
        print('last loss :',ls[-1])
    if(train):
        if(seed != 0):
            training_data, seed = load(sgm, seed)
        else:
            training_data, seed = load(sgm)
    else:
        training_data = Dataset([0],[0],[0])
    bs = (int) (len(training_data) if len(training_data) > 1 else 1)
    train_dl = DataLoader(training_data, batch_size = bs, shuffle = True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(u_net.parameters(), lr = 0.00025)
    recup = 0
    epochs = 6000 - recup
    loss_epoch = 0
    for epoch in range(int(epochs)):
        if(not train):
            break
        x, y = next(iter(train_dl))
        optimizer.zero_grad()
        output = u_net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        ls.append(loss_epoch/bs)
        loss_epoch = 0
        sv = str(epoch + 1 + recup)
        print('loss', round(ls[-1], 3), sv+'/'+str(epochs + recup), str(round((epoch + recup + 1) / (epochs + recup), 3) * 100)+'%')
        if(save):
            torch.save({'model':u_net.state_dict(),
                        'loss':ls,
                        'seed':seed},
                       'save/name'+sv+'.pth')
            print('saved:', ('save/name'+sv+'.pth'))
    
    path = 'Data/train/152.npy' # The image we want 
    
    img = (torch.from_numpy(np.load(path))[None, :]).float()
    z = u_net(img[None, :])
    path = 'Data/train/label'+str('152')+'.npy'
    data = torch.from_numpy(Blur.blur_gaussian(img.detach().numpy()[0], sgm))
    #data = torch.from_numpy(Blur.blur_land(img.detach().numpy()[0]))
    
    if(difference):
        dif = (z[0].detach()[0].numpy() - data.detach().numpy())
        f, ax = plt.subplots(1, 3)
        with torch.no_grad():
            ax[0].imshow(img[0].numpy(), cmap = 'gray')
            ax[1].imshow(z[0][0].numpy(), cmap = 'gray')
            ax[2].imshow(data.numpy(), cmap = 'gray')
            matplotlib.image.imsave('target.png', z[0][0].numpy(), cmap='gray')
            matplotlib.image.imsave('label.png', data.numpy(), cmap='gray')
    
    plt.subplots()
    
    plt.plot(ls)
    
    plt.show()
    
    if(difference):
        cmn = cm.get_cmap('RdBu_r', 500)
        newcolors = cmn(np.linspace(0, 1, 500))
        pink = np.array([255/255, 25/255, 150/255, 1])
        p = 5
        gap = (dif.max() - dif.min()) / 500
        newcolors[(int) (-dif.min() / gap) - p: (int) (-dif.min() / gap) + p, :] = pink
        newcmp = ListedColormap(newcolors)
        fig = plt.figure(frameon = False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        plt.gca().invert_yaxis()
        plt.pcolor(dif,
                   norm=colors.Normalize(),
                   cmap=newcmp)
        plt.colorbar()
        plt.savefig('difference.png')
        plt.show()
        print('mean:',np.mean(dif))
        print('scale:',dif.min(), dif.max())
    
    valids = []
    save_max = 201 # Number of saves
    
    for j in range(1, save_max):
        if(not valid):
            break
        checkpoint = torch.load('save/name'+str(j)+'.pth')
        u_net.load_state_dict(checkpoint['model'])
        y = u_net(img[None, :])
        valids.append(np.sqrt(criterion(y[0], data).item()))
    
    if(valid):
        plt.plot(valids)
        
        plt.show()

train(u_net)