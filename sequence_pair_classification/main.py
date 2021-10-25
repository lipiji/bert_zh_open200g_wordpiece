import sys
sys.path.append('..')
import torch
from torch import nn
import torch.nn.functional as F
import random
import os
import argparse
#### Load pretrained bert model
from bert import BERTLM
from data import Vocab, CLS, SEP, MASK
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--number_class', type = int)
    parser.add_argument('--number_epoch', type = int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--model_save_path', type=str)
    return parser.parse_args()


def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path, map_location='cpu')
    bert_name = args.bert_path.split("/")[-1]
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    bert_model = bert_model.cuda(device)
    return bert_model, bert_vocab, bert_args, bert_name

def ListsToTensor(xs, vocab):
    batch_size = len(xs)
    lens = [ len(x1) + len(x2) + 2 for x1, x2 in xs]
    mx_len = max(lens)
    ys = []
    for i, (x1, x2) in enumerate(xs):
        y =  vocab.token2idx([CLS]+x1+[SEP]+x2) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data

def batchify(data, vocab):
    return ListsToTensor(data, vocab)

class myModel(nn.Module):
    def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, vocab):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.vocab = vocab
        self.fc = nn.Linear(self.embedding_size, self.num_class)
    
    def forward(self, data, fine_tune=False):
        # size of data is [batch_max_len, batch_size]
        batch_size = len(data)
        data = batchify(data, self.vocab)
        data = data.cuda(self.device)
        x = self.bert_model.work(data)[1].cuda(self.device)
        if not fine_tune:
            x = x.detach()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view(batch_size, self.embedding_size)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    args = parse_config()
    
    directory = args.model_save_path
    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 
    # Data Preparation
    train_path, dev_path = args.train_data, args.dev_data
    mydata = DataLoader(train_path, dev_path, args.max_len)
    print ('data is ready')

    # myModel construction
    print ('Initializing model...')
    bert_vocab = args.bert_vocab
    bert_model, bert_vocab, bert_args, bert_name = init_bert_model(args, args.gpu_id, bert_vocab)
    batch_size = args.batch_size
    number_class = args.number_class
    embedding_size = bert_args.embed_dim
    fine_tune = args.fine_tune
    model = myModel(bert_model, number_class, embedding_size, batch_size, args.dropout, args.gpu_id, bert_vocab)
    model = model.cuda(args.gpu_id)

    print ('Model construction finished.')

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    #--- training part ---#
    num_epochs = args.number_epoch
    training_data_num, dev_data_num = mydata.train_num, mydata.dev_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = int(dev_data_num / batch_size) + 1
    max_dev_acc = 0.0

    for epoch in range(num_epochs):
        loss_accumulated = 0.
        model.train()
        print ('-------------------------------------------')
        if epoch % 5 == 0:
            print ('%d epochs have run' % epoch)
        else:
            pass
        total_train_pred = list()
        total_train_true = list()
        batches_processed = 0
        for train_step in range(train_step_num):
            batches_processed += 1
            optimizer.zero_grad()

            train_batch_text_list, train_batch_label_list = mydata.get_next_batch(batch_size, mode = 'train')
            train_batch_output = model(train_batch_text_list, fine_tune)
            train_batch_label = torch.tensor(np.array(train_batch_label_list)).cuda(args.gpu_id)

            ce_loss = criterion(train_batch_output, train_batch_label)
            loss_accumulated += ce_loss.item()

            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            one_train_result_detached = train_batch_output.detach()#numpy()
            one_train_result = one_train_result_detached.cpu().numpy()
            for instance in one_train_result:
                total_train_pred.append(instance)
            total_train_true.extend(train_batch_label_list)

            if batches_processed % args.print_every == 0:
                print ("Batch %d, loss %.5f" % (batches_processed, loss_accumulated / batches_processed))

            if train_step == train_step_num - 1:
                train_pred_list = np.argmax(total_train_pred, axis = 1)
                assert len(train_pred_list) == len(total_train_true)
                train_acc = accuracy_score(total_train_true, train_pred_list)
                print ('At epoch %d, training accuracy is %f' % (epoch, train_acc * 100))
        model.eval()
        with torch.no_grad():
            total_dev_pred = list()
            total_dev_true = list()

            for dev_step in range(dev_step_num):
                dev_batch_text_list, dev_batch_label_list = mydata.get_next_batch(batch_size, mode = 'dev')
                dev_batch_output = model(dev_batch_text_list, fine_tune = False)
                one_dev_result = dev_batch_output.cpu().numpy()
                for dev_instance in one_dev_result:
                    total_dev_pred.append(dev_instance)
                total_dev_true.extend(dev_batch_label_list)

            valid_dev_result = total_dev_pred[:dev_data_num]
            valid_dev_pred = np.argmax(valid_dev_result, axis = 1)
            valid_dev_true = total_dev_true[:dev_data_num]
            assert len(valid_dev_pred) == len(valid_dev_true)
            dev_acc = accuracy_score(valid_dev_true, valid_dev_pred)
            if dev_acc > max_dev_acc:
                torch.save({'args':args, 'model':model.state_dict(), 
                        'bert_args': bert_args, 
                        'bert_vocab':bert_vocab
                        }, directory + '/epoch_%d_dev_acc_%.4f_%s'%(epoch + 1, dev_acc, bert_name))
                max_dev_acc = dev_acc
            print ('At epoch %d dev accuracy is %f' % (epoch, dev_acc * 100))

    print ('-----------------------------------------------------')
    print ('At this run, the maximum accuracy is %f' % max_dev_acc * 100)
    print ('-----------------------------------------------------')
