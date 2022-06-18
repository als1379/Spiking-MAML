from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.clock_driven import neuron, functional, encoding, surrogate, layer
import sys
import time
import numpy as np
from tqdm import tqdm
from ann import ANN
from snn import SNN
from    omniglotNShot import OmniglotNShot
import  argparse

_seed_ =  2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


def main(args):
    #Parameters Setting
    device = "cuda:0" 
    dataset_dir = "./"
    batch_size = 100  
    learning_rate = 1e-4
    T = args.T
    train_epoch = args.epoch
    log_dir = "./"
    model_dir="path to ANN and SNN saved models on your local machine or on your Google Drive"
    #MAML
    update_step = args.update_step
    update_lr = args.update_lr
    # #Data transormations
    # test_list_transforms = [
    #     transforms.ToTensor(),
    # ]
    
    # train_list_transforms = [
    #     transforms.RandomCrop(26),
    #     transforms.Pad(1),
    #     transforms.ToTensor(),
    # ]
    
    # #Data loaders
    # train_transform = transforms.Compose(train_list_transforms)
    # test_transform = transforms.Compose(test_list_transforms)

    # train_data_loader = torch.utils.data.DataLoader(
    #     dataset=torchvision.datasets.FashionMNIST(
    #         root=dataset_dir,
    #         train=True,
    #         transform=train_transform,
    #         download=True),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True)
    # test_data_loader = torch.utils.data.DataLoader(
    #     dataset=torchvision.datasets.FashionMNIST(
    #         root=dataset_dir,
    #         train=False,
    #         transform=test_transform,
    #         download=True),
    #     batch_size=batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     pin_memory=True)

    db_train = OmniglotNShot('omniglot',
                    batchsz=args.task_num,
                    n_way=args.n_way,
                    k_shot=args.k_spt,
                    k_query=args.k_qry,
                    imgsz=args.imgsz)

    #Building or loading models (ANN and SNN)
    print('Load pretrained model? (y/n) ')
    pretrained=input()
    if pretrained=='y':
      print('Loading... ')
      ann=torch.load(model_dir+'/ANN_Params.pt',map_location=device)
      snn=torch.load(model_dir+'/SNN_Params.pt',map_location=device)
      print('Pretrained model loaded!')
      print('Evaluation on test data:')
      train_epoch=0

    else: 
      ann = ANN().to(device)
      snn = SNN(T=T).to(device)
      print('Model initialized with random weights!')
    
       

    # Weight Sharing: set ptr of snn's param to point ann's param
    params_ann = ann.named_parameters()
    params_snn = snn.named_parameters()
    dict_params_snn = dict(params_snn)
    for name, param in params_ann:
        if name in dict_params_snn:
            dict_params_snn[name].data = param.data


            
    #Optimizer Settings        
    optimizer_ann = torch.optim.Adam(ann.parameters(), lr=learning_rate, betas=(0.8, 0.99), eps=1e-08, weight_decay=1e-06)
    # criterion = nn.CrossEntropyLoss()
    
    #Learning 
    print('Learning started...')
    for epoch in range(train_epoch):
        ann.train()
        snn.train()
        if epoch>=1:
            for m in ann.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for m in snn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        correct_ann = 0
        correct_snn = 0
        sample_num = 0
        t_start = time.perf_counter()
        # MAML
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                    torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        losses = [0 for _ in range(update_step)]
        corrects_ann = [0 for _ in range(update_step)]
        corrects_snn = [0 for _ in range(update_step)]

        for i in range(task_num):    
            x_train = x_spt[i]
            y_train = y_spt[i]
            x_test = x_qry[i]
            y_test = y_qry[i]
            fast_weights = ann.parameters()
            print(len(fast_weights))
            # label_one_hot = F.one_hot(y, 10).float()
            # label_one_hot_test = F.one_hot(y_test, 10).float()
            optimizer_ann.zero_grad()
            for k in range(update_step):
                
                # MAML task train
                outputs = ann(x_train, fast_weights)#ANN output

                with torch.no_grad():
                    out_spikes_counter = snn(x_train, fast_weights)#SNN spike counts

                # out_spikes_counter_frequency = out_spikes_counter   
                outputs.data.copy_(out_spikes_counter)#Replacing SNN output in ANN output layer

                loss = F.cross_entropy(outputs, y_train)# Comuting the loss in ANN (by the SNN output)

                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, fast_weights)))
                print("************************************************")
                print(len(fast_weights))
                functional.reset_net(snn)

                # MAML task test
                outputs_test = ann(x_test, fast_weights)
                with torch.no_grad():
                    pred_ann = F.softmax(outputs_test, dim=1).argmax(dim=1)
                    correct_ann = torch.eq(pred_ann, y_test).sum().item()  # convert to numpy
                    corrects_ann[k]+= correct_ann
                with torch.no_grad():
                    out_spikes_counter_test = snn(x_test, fast_weights)#SNN spike counts
                outputs_test.data.copy_(out_spikes_counter_test)
                
                loss_test = F.cross_entropy(outputs_test, y_test)
                losses[k] += loss_test

                with torch.no_grad():
                    pred_snn = F.softmax(outputs_test, dim=1).argmax(dim=1)
                    correct_snn = torch.eq(pred_snn, y_test).sum().item()  # convert to numpy
                    corrects_snn[k] += correct_snn

                functional.reset_net(snn)

        
        # end of all tasks
        # sum over all losses on query set across all tasks    
        final_loss = losses[-1] / task_num
        final_loss.backward()#computing the gradients in ANN
        optimizer_ann.step()#updating the shared weights
        
        acc_ann = np.array(corrects_ann) / (querysz * task_num)
        acc_snn = np.array(corrects_snn) / (querysz * task_num)
        print(f'epoch={epoch}, train_ann={acc_ann}, train_snn={acc_snn}') #, t_train={t_train}, t_test={t_test}')
        t_train = time.perf_counter() - t_start

        #Evaluation on test samples
    #     ann.eval()
    #     snn.eval()
    #     t_start = time.perf_counter()          
    #     with torch.no_grad():
    #         correct_snn = 0
    #         correct_ann = 0
    #         sample_num = 0
    #         for img, label in tqdm(test_data_loader, position=0):
    #             img = img.to(device)
    #             label = label.to(device)
    #             predict_ann = ann(img)
    #             correct_ann += (predict_ann.argmax(1) == label).sum().item()
    #             sample_num += label.numel()
    #             out_spikes_counter_frequency = snn(img)
    #             correct_snn += (out_spikes_counter_frequency.argmax(1) == label).sum()

    #             functional.reset_net(snn)

    #         acc_ann = correct_ann / sample_num
    #         acc_snn = correct_snn / sample_num
    #         t_test = time.perf_counter() - t_start
            
    #         print(f'epoch={epoch}, acc_ann={acc_ann}, acc_snn={acc_snn}') #, t_train={t_train}, t_test={t_test}')
    # ann.eval()
    # snn.eval()
    # with torch.no_grad():
    #     correct_snn = 0
    #     correct_ann = 0
    #     sample_num = 0
    #     for img, label in tqdm(test_data_loader, position=0):
    #         img = img.to(device)
    #         label = label.to(device)
    #         predict_ann = ann(img)
    #         correct_ann += (predict_ann.argmax(1) == label).sum().item()
    #         sample_num += label.numel()
    #         out_spikes_counter_frequency = snn(img)
  
    #         # correct_snn += (out_spikes_counter_frequency.argmax(1) == label).sum()
    #         lab = F.one_hot(label, 10).float()
    #         lab = (lab.cpu()).numpy()
    #         lab = lab.astype(bool)
    #         out_spikes_counter_frequency = ((out_spikes_counter_frequency.cpu()).detach()).numpy()
    #         lab2 = out_spikes_counter_frequency[lab]
    #         correct_snn += (out_spikes_counter_frequency.max(1) == lab2).sum()
            
    #         functional.reset_net(snn)

    #     acc_ann = correct_ann / sample_num
    #     acc_snn = correct_snn / sample_num
    #     print(f' Final Result: Acc_ANN={acc_ann}, Acc_SNN={acc_snn}') #, t_train={t_train}, t_test={t_test}')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=40)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--T', type=int, help='neurons time', default=4)
    argparser.add_argument('--threshold', type=float, help='IF neurons threshold', default=1.0)

    args = argparser.parse_args()

    main(args)

