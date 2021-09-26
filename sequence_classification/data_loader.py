import sys
sys.path.append('..')
from google_bert import BasicTokenizer
import random

class DataLoader:
    def __init__(self, train_path, dev_path, max_len):
        self.tokenizer = BasicTokenizer()
        self.train_path = train_path
        self.dev_path = dev_path
        self.max_len = max_len 

        self.train_seg_list, self.train_tgt_list = self.load_data(train_path)
        self.dev_seg_list, self.dev_tgt_list = self.load_data(dev_path)

        self.train_num, self.dev_num = len(self.train_seg_list), len(self.dev_seg_list)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))

        self.train_idx_list, self.dev_idx_list = [i for i in range(self.train_num)], [j for j in range(self.dev_num)]
        self.shuffle_train_idx()

        self.train_current_idx = 0
        self.dev_current_idx = 0

    def load_data(self, path):
        src_list = list() # src_list contains segmented text
        tgt_list = list() # tgt_list contains class number
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                content_list = l.strip('\n').split('\t')
                text = content_list[0]
                target = int(content_list[1])
                seg_text = self.tokenizer.tokenize(text)
                src_list.append(self.seq_cut(seg_text))
                tgt_list.append(target)
        return src_list, tgt_list

    def shuffle_train_idx(self):
        random.shuffle(self.train_idx_list)

    def seq_cut(self, seq):
        if len(seq) > self.max_len:
            seq = seq[ : self.max_len]
        return seq

    def get_next_batch(self, batch_size, mode):
        batch_text_list, batch_label_list = list(), list()
        if mode == 'train':
            if self.train_current_idx + batch_size < self.train_num - 1:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                    batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                self.train_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.train_current_idx + i
                    if curr_idx > self.train_current_idx - 1:
                        self.shuffle_train_idx()
                        curr_idx = 0
                        batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                        batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                    else:
                        batch_text_list.append(self.train_seg_list[self.train_idx_list[curr_idx]])
                        batch_label_list.append(self.train_tgt_list[self.train_idx_list[curr_idx]])
                self.train_current_idx = 0

        elif mode == 'dev':
            if self.dev_current_idx + batch_size < self.dev_num - 1:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    batch_text_list.append(self.dev_seg_list[curr_idx])
                    batch_label_list.append(self.dev_tgt_list[curr_idx])
                self.dev_current_idx += batch_size
            else:
                for i in range(batch_size):
                    curr_idx = self.dev_current_idx + i
                    if curr_idx > self.dev_num - 1: # 对dev_current_idx重新赋值
                        curr_idx = 0
                        self.dev_current_idx = 0
                    else:
                        pass
                    batch_text_list.append(self.dev_seg_list[curr_idx])
                    batch_label_list.append(self.dev_tgt_list[curr_idx])
                self.dev_current_idx = 0
        else:
            raise Exception('Wrong batch mode!!!')

        return batch_text_list, batch_label_list
