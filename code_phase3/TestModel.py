#!/usr/bin/env python
# coding: utf-8

# ### Load Packages

# In[2]:


import torch
import time
from DataLoader import load_dataset
from ConvNet import get_model
import numpy as np
import cv2
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


# ### Keep checking learning rate

# In[3]:


def check_lr(optimizer, epoch):
    if epoch >= 10:
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.1


# ### Copy pretrained model

# In[4]:


def copy_pretrain_vals(model, vgg_dict, isstrict):
    model.load_state_dict(vgg_dict,strict=isstrict)
    return model


# ### Generating a file list array for data loader

# In[5]:


def get_files_list(fg_file, bg_file):
    fg_list = []
    bg_list = []
    with open(fg_file, 'r') as f:
        fg_list = f.readlines()
    with open(bg_file, 'r') as f:
        bg_list = f.readlines()
    for i in range(len(fg_list)):
        fg_list[i] = fg_list[i].strip('\n').strip('\r')
    for i in range(len(bg_list)):
        bg_list[i] = bg_list[i].strip('\n').strip('\r')
    arr = []
    cnt = 0
    for i in range(len(fg_list)):
        for j in range(20):
            arr.append((fg_list[i], bg_list[cnt], fg_list[i], fg_list[i].strip('.png') + '_' + str(j) + '.png'))
            cnt += 1
    return arr


# ### Test model function

# In[6]:


def test_model(model, dataset, trimap_path, cuda=False):
    model.eval()
    mse_diffs = 0.
    sad_diffs = 0.
    for iteration in range(dataset.__len__()):
        torch.cuda.empty_cache()
        file_info =  dataset.__getinfo__(iteration)
        print(file_info)
        batch = dataset.__getitem__(iteration)
        if batch == None:
            continue
#         print('oompa')
#         print(batch[0].shape)
        batch0 = batch[0].reshape(1, batch[0].shape[0], batch[0].shape[1], batch[0].shape[2])
        batch1 = batch[1].reshape(1, batch[1].shape[0], batch[1].shape[1], batch[1].shape[2])
        batch2 = batch[2].reshape(1, batch[2].shape[0], batch[2].shape[1], batch[2].shape[2])
        batch3 = batch[3].reshape(1, batch[3].shape[0], batch[3].shape[1], batch[3].shape[2])
        batch4 = batch[4].reshape(1, batch[4].shape[0], batch[4].shape[1], batch[4].shape[2])
        img = Variable(batch0)
        alpha = Variable(batch1)
        fg = Variable(batch2)
        bg = Variable(batch3)
#         trimap = Variable(batch4)
        trimap = cv2.imread(trimap_path + file_info[3])[:, :, 0]
#         cv2.imwrite('trimap.jpg', trimap)
#         print('########')
#         print(img.shape)
        trimap = cv2.resize(trimap, (img.shape[3], img.shape[2]), interpolation=cv2.INTER_LINEAR)
#         print(trimap.shape)
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
#         print(trimap.size())
        trimap = trimap.reshape(1, trimap.shape[0], trimap.shape[1], trimap.shape[2])
#         print(trimap.size())
        trimap = Variable(trimap)
        if cuda:
            img = img.cuda()
            alpha = alpha.cuda()
            fg = fg.cuda()
            bg = bg.cuda()
            trimap = trimap.cuda()

        pred_alpha = model(torch.cat((img, trimap), 1))
        pred_alpha = pred_alpha.data
        if cuda:
            pred_alpha = pred_alpha.cpu()
        pred_alpha = pred_alpha.numpy()[0, 0, :, :]
        pred_alpha[trimap == 255] = 1.
        pred_alpha[trimap == 0 ] = 0.
        pixel = float((trimap == 128).sum())
        alpha = cv2.imread('/scratch/matting/dataset/Test_set/adobe/alpha/' + file_info[2])[:, :, 0]
        alpha = cv2.resize(alpha, (pred_alpha.shape[1], pred_alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('alpha.jpg', alpha)
        alpha = alpha / 255.
        print('############')
        print(pred_alpha.shape, alpha.shape)
        mse_diff = ((pred_alpha - alpha) ** 2).sum() / pixel
        sad_diff = np.abs(pred_alpha - alpha).sum()
        mse_diffs += mse_diff
        sad_diffs += sad_diff
        print(mse_diffs/(iteration+1), sad_diffs/(iteration+1), iteration)
        pred_alpha = (pred_alpha * 255).astype(np.uint8)
        print(pred_alpha)
        cv2.imwrite('alpha_pred.png', pred_alpha)
        plt.imshow(pred_alpha)
        if iteration == 40:
            break


# ### Save model

# In[7]:


def save_model(path, model):
    torch.save(model, path)


# ### Main function

# In[8]:


def main(cuda=False):
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    if cuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    fg_path = '/scratch/matting/dataset/Test_set/adobe/fg/'
    bg_path = '/scratch/matting/dataset/VOCO_dataset/'
    alpha_path = '/scratch/matting/dataset/Test_set/adobe/alpha/'
    trimap_path = '/scratch/matting/dataset/Test_set/adobe/trimaps/'
#     vgg_path = '/scratch/matting/dataset/full_trained.pth'
    vgg_path = '/scratch/matting/model_1.pth'
    fg_file = '/scratch/matting/dataset/Test_set/test_fg_names.txt'
    bg_file = '/scratch/matting/dataset/Test_set/test_bg_names.txt'
    files_list = get_files_list(fg_file, bg_file)
#     print(files_list)
    dataset = load_dataset(fg_path, alpha_path, bg_path, files_list)
    model = get_model()
    vgg_dict = torch.load(vgg_path, map_location='cpu')
#     vgg_dict = torch.load_state_dict(vgg_dict)
#     model = copy_pretrain_vals(model, vgg_dict["state_dict"], True)
    model = copy_pretrain_vals(model, vgg_dict.state_dict(), True)
    print(model)
    if cuda:
        model = model.cuda()
    test_model(model, dataset, trimap_path, cuda)
main()


# In[ ]:




