#!/usr/bin/env python
# coding: utf-8

# ### Load Packages

# In[1]:


import import_ipynb
# import Dataloader
import torch
import time
from DataLoader import load_dataset
from ConvNet import get_model
import torch.optim as optim
from torch.autograd import Variable


# ### Encoder Decoder Loss function

# In[2]:


def loss_stage1(alpha, trimap, pred_alpha, fg, bg, img, cuda):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi
    t3_wi = torch.cat((wi, wi, wi), 1)
    if cuda:
        t3_wi = t3_wi.cuda()
        t_wi = wi.cuda()

    unknown_region_size = t_wi.sum()

    # alpha loss
    alpha = alpha / 255.
    alpha_loss = torch.sqrt((pred_alpha - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)

    # composite rgb loss
    pred_alpha_3 = torch.cat((pred_alpha, pred_alpha, pred_alpha), 1)
#     print(fg.shape, bg.shape, pred_alpha_3.shape)
    comp = pred_alpha_3 * fg + (1. - pred_alpha_3) * bg
    comp_loss = torch.sqrt((comp - img) ** 2 + 1e-12) / 255.
    comp_loss = (comp_loss * t3_wi).sum() / (unknown_region_size + 1.) / 3.
    return alpha_loss, comp_loss


# ### Refinement stage Loss function

# In[3]:


def loss_stage2(alpha, pred_alpha, trimap, cuda):
    wi = torch.zeros(trimap.shape)
    wi[trimap == 128] = 1.
    t_wi = wi
    if cuda:
        t_wi = wi.cuda()
    unknown_region_size = t_wi.sum()
    # alpha loss
    alpha = alpha / 255.
#     print(pred_alpha.size())
    alpha_loss = torch.sqrt((pred_alpha - alpha)**2 + 1e-12)
    alpha_loss = (alpha_loss * t_wi).sum() / (unknown_region_size + 1.)
    
    return alpha_loss


# ### Keep checking learning rate

# In[4]:


def check_lr(optimizer, epoch):
    if epoch >= 10:
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.1


# ### Copy pretrained model

# In[5]:


def copy_pretrain_vals(model, vgg_dict):
    model.load_state_dict(vgg_dict,strict=False)
    return model


# ### Generating a file list array for data loader

# In[6]:


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
        for j in range(100):
            arr.append((fg_list[i], bg_list[cnt], fg_list[i]))
            cnt += 1
    return arr


# ### Train model function

# In[7]:


def train_model(stage, model, optimizer, dataset, epoch, cuda=False):
    model.train()
#     for iteration, batch in enumerate(dataset, 1):
#         print(iteration)
#         torch.cuda.empty_cache()
#         print(iteration, epoch)
#         print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape)
#         img = Variable(batch[0])
#         alpha = Variable(batch[1])
#         fg = Variable(batch[2])
#         bg = Variable(batch[3])
#         trimap = Variable(batch[4])

#         if cuda:
#             img = img.cuda()
#             alpha = alpha.cuda()
#             fg = fg.cuda()
#             bg = bg.cuda()
#             trimap = trimap.cuda()

#         check_lr(optimizer, epoch)
#         optimizer.zero_grad()

#         pred_alpha = model(torch.cat((img, trimap), 1))

#         alpha_loss, comp_loss = loss_stage1(alpha, trimap, pred_alpha, fg, bg, img, cuda)
#         loss = alpha_loss*0.5 + comp_loss*0.5
#         loss.backward()
#         optimizer.step()
    print(dataset.__len__())
    for iteration in range(dataset.__len__()):
#         print(iteration)
        torch.cuda.empty_cache()
        try:
            batch = dataset.__getitem__(iteration)
            if batch == None:
                continue
    #         print(batch[0].shape)
    #         print(batch[0])
            batch0 = batch[0].reshape(1, batch[0].shape[0], batch[0].shape[1], batch[0].shape[2])
            batch1 = batch[1].reshape(1, batch[1].shape[0], batch[1].shape[1], batch[1].shape[2])
            batch2 = batch[2].reshape(1, batch[2].shape[0], batch[2].shape[1], batch[2].shape[2])
            batch3 = batch[3].reshape(1, batch[3].shape[0], batch[3].shape[1], batch[3].shape[2])
            batch4 = batch[4].reshape(1, batch[4].shape[0], batch[4].shape[1], batch[4].shape[2])
            img = Variable(batch0)
            alpha = Variable(batch1)
            fg = Variable(batch2)
            bg = Variable(batch3)
            trimap = Variable(batch4)
    #         print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape, batch[4].shape)
    #         print(img.shape, alpha.shape, bg.shape, fg.shape, trimap.shape)

            if cuda:
                img = img.cuda()
                alpha = alpha.cuda()
                fg = fg.cuda()
                bg = bg.cuda()
                trimap = trimap.cuda()

            check_lr(optimizer, epoch)
            optimizer.zero_grad()

            pred_mattes, pred_alpha = model(torch.cat((img, trimap), 1))

    #         alpha_loss, comp_loss = loss_stage1(alpha, trimap, pred_alpha, fg, bg, img, cuda)
    #         loss = alpha_loss*0.5 + comp_loss*0.5
            wl_weight = 0.5

            if stage == 1:
                # stage1 loss
                alpha_loss, comp_loss = loss_stage1(alpha,trimap,pred_mattes, fg, bg, img , cuda)
                loss = alpha_loss * wl_weight + comp_loss * (1. - wl_weight)
            elif stage == 2:
                # stage2 loss
                loss = loss_stage2(alpha, pred_alpha, trimap, cuda)
            else:
                # stage3 loss = stage1 loss + stage2 loss
                alpha_loss, comp_loss = loss_stage1(alpha,trimap,pred_mattes, fg, bg, img , cuda)
                loss1 = alpha_loss * wl_weight + comp_loss * (1. - wl_weight)
                loss2 = loss_stage2(alpha, pred_alpha, trimap, cuda)
                loss = loss1 + loss2


            loss.backward()
            optimizer.step()
            if iteration % 20 == 0:
                print(iteration, epoch, loss)
        except:
            print(iteration)


# ### Save model

# In[8]:


def save_model(path, model):
    torch.save(model, path)


# ### Main function

# In[ ]:


def main(stage,cuda=True):
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    if cuda:
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    
    fg_path = '/scratch/matting/dataset/Training_set/adobe/fg/'
    bg_path = '/scratch/matting/dataset/train2014/'
    alpha_path = '/scratch/matting/dataset/Training_set/adobe/alpha/'
    vgg_path = '/scratch/matting/dataset/vgg_state_dict.pth'
    stage1_path = '/scratch/matting/dataset/full_trained.pth'
    fg_file = '/scratch/matting/dataset/Training_set/training_fg_names.txt'
    bg_file = '/scratch/matting/dataset/Training_set/training_bg_names.txt'
    epochs = 10
    files_list = get_files_list(fg_file, bg_file)
    dataset = load_dataset(fg_path, alpha_path, bg_path, files_list)
    model = get_model(stage)
    print(model)
#     vgg_dict = torch.load(vgg_path)
#     model = copy_pretrain_vals(model, vgg_dict)
    stage1_model = torch.load(stage1_path, map_location='cpu')['state_dict']
    model = copy_pretrain_vals(model, stage1_model)
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    for i in range(epochs):
        train_model(stage ,model, optimizer, dataset, i, cuda)
        save_model('/scratch/matting/model_' + str(i) + '_stage2.pth', model)
main(2)

