import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
import argparse
from sklearn import metrics
from dataloader import AGNEWS_Dataset
from model import charCNN

wandb.init(project="charcnn")

parser = argparse.ArgumentParser(description='Specify dataset and model variant (small/large)')
parser.add_argument('--model-size', default='large',
                    help='Model size. Note: Number of 1d conv kernels in each \
                     layer is 256 for small and 1024 for large. Size of fully \
                     connected layer is 1024 for small and 2048 for large')
parser.add_argument('--dataset', default='ag_news',
                    help='Choose among [ag_news, \
                               sogou_news, \
                               dbpedia, \
                               yelp_review_full, \
                               yelp_review_polarity, \
                               amazon_review_full, \
                               amazon_review_polarity, \
                               yahoo_answers]')
parser.add_argument('--max-seq-len', type=int, default=1014,
                    help='Maximum length of character sequence given as input \
                    to the ConvNet')
parser.add_argument('--alpha-path', default='alphabet.json',help='Path to \
                    alphabet file')
parser.add_argument('--batch-size', type=int, default=128,help='Size of \
                    mini-batch')
parser.add_argument('--num-characters',type=int, default=70,help='Number of \
                    characters in the alphabet')
parser.add_argument('--lrs-milestones',default='3,6,9,12,15,18,21,24,27,30',
                    help='Milestones for the learning rate scheduler to decay \
                    learning rate. Format: milestone1,milestone2, up to the \
                    last milestone')
parser.add_argument('--total-epochs',type=int,default=200,help='Total number \
                    of epochs for training')
parser.add_argument('--resume', action='store_const',const=1,help='Use if \
                    resuming training from checkpoint file')
parser.add_argument('--start-epoch',type=int,default=1,help='First epoch \
                    number')
parser.add_argument('--save-every',type=int,default=4,help='How often the \
                    model is saved. NOTE: Not used since we use validation \
                    accuracy')
parser.add_argument('--print-every',type=int,default=1,help='How often \
                    training stats are printed')
parser.add_argument('--max-norm',type=int,default=400,help='Max value used for \
                    gradient clipping')
parser.add_argument('--model-name',default='bestmodel.ckpt',help='Name of \
                    checkpoint file')

def get_lr(optimizer):
    """
    Gets the current learning rate from the optimizer during training
    @params optimizer (torch.optim.Optimizer): Optimizer object
    @returns param_group['lr'] (int): Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

datasets = ['ag_news',
           'sogou_news',
           'dbpedia',
           'yelp_review_full',
           'yelp_review_polarity',
           'amazon_review_full',
           'amazon_review_polarity',
           'yahoo_answers']

model_params = {
    'small':{'fc_size':1024,'conv_channel_size':256,'mean':0,'std':0.05},
    'large':{'fc_size':2048,'conv_channel_size':1024,'mean':0,'std':0.02}
    }

dataset_info = {'ag_news':
         {'train_path':'./ag_news_csv/train.csv',
          'test_path':'./ag_news_csv/test.csv',
          'num_classes':4},
         'sogou_news':
         {'train_path':'./sogou_news_csv/train.csv',
          'test_path':'./sogou_news_csv/test.csv',
          'num_classes':5},
         'dbpedia':
         {'train_path':'./dbpedia_csv/train.csv',
          'test_path':'./dbpedia/test.csv',
          'num_classes':14},
         'yelp_review_full':
         {'train_path':'./yelp_review_full_csv/train.csv',
          'test_path':'./yelp_review_full_csv/test.csv',
          'num_classes':5},
         'yelp_review_polarity':
         {'train_path':'./yelp_review_polarity_csv/train.csv',
          'test_path':'./yelp_review_polarity_csv/test.csv',
          'num_classes':2},
         'amazon_review_full':
         {'train_path':'./amazon_review_full_csv/train.csv',
          'test_path':'./amazon_review_full_csv/test.csv',
          'num_classes':5},
         'amazon_review_polarity':
         {'train_path':'./amazon_review_polarity_csv/train.csv',
          'test_path':'./amazon_review_polarity_csv/test.csv',
          'num_classes':2},
         'yahoo_answers':
         {'train_path':'./yahoo_answers_csv/train.csv',
          'test_path':'./yahoo_answers_csv/test.csv',
          'num_classes':10}
         }

train_path = dataset_info[dataset]['train_path']
test_path = dataset_info[dataset]['test_path']
num_classes = dataset_info[dataset]['num_classes']
fc_size = model_params[model_size]['fc_size']
conv_channel_size = model_params[model_size]['conv_channel_size']
model_path = os.path.join('./models',dataset+'_'+args.model_size)
loss_history = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ag_dataset_train = AGNEWS_Dataset(train_path,args.alpha_path,args.max_seq_len)
dataloader_train = DataLoader(ag_dataset_train,batch_size=128,shuffle=True,num_workers=4)
ag_dataset_test = AGNEWS_Dataset(test_path,alpha_path,max_seq_len)
dataloader_test = DataLoader(ag_dataset_test,batch_size=128,shuffle=True,num_workers=4)


model = charCNN(args.num_characters,conv_channel_size,fc_size,num_classes,
                args.max_seq_len,model_params[model_size]['mean'],model_params[model_size]['std'])
wandb.watch(model)

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


if resume:
    checkpoint = torch.load(os.path.join(model_path,args.model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    print('Loaded model from checkpoint ...')

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, list(args.milestones), gamma=0.5, last_epoch=-1)

acc_history = [0]

for epoch in range(args.start_epoch,args.total_epochs+1):
    batch_loss_history = []
    for i,(char_seq,cls) in enumerate(dataloader_train):

        cls = torch.LongTensor(cls)

        char_seq = char_seq.to(device)
        cls = cls.to(device)
        optimizer.zero_grad()

        outputs = model(char_seq)
        loss = F.nll_loss(outputs,cls-1)
        batch_loss_history.append(loss.item())
        loss_history.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

    scheduler.step()

    acc = 0
    batches = 0
    for i,(char_seq,cls) in enumerate(dataloader_test):
        batches += 1
        cls = torch.LongTensor(cls)

        char_seq = char_seq.to(device)
        cls = cls.to(device)

        outputs = model(char_seq)
        predicted_cls = outputs.max(1)[1]
        acc += metrics.accuracy_score((cls-1).cpu().numpy(),predicted_cls.cpu().numpy())
    avg_accuracy = acc/batches


    if epoch == args.start_epoch or avg_accuracy > max(acc_history):
        torch.save(
            {
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'epoch':epoch+1,
                'loss_history':loss_history
            },
            os.path.join(model_path,'bestmodel.ckpt')
            )
        print('Saved model ...')

    acc_history.append(avg_accuracy)

    if epoch % args.print_every == 0:
        mean_loss_per_epoch = sum(batch_loss_history)/len(batch_loss_history)
        print('[{}/{}] Loss: {}'.format(epoch,total_epochs,mean_loss_per_epoch))
        wandb.log({"Train Loss": mean_loss_per_epoch,"Learning Rate": get_lr(optimizer)})
        print('Accruacy: ',avg_accuracy)
        print('Test error: ',1-avg_accuracy)
        wandb.log({"Test Accuracy": avg_accuracy,"Test Error": 1-avg_accuracy})



print('Training complete')
