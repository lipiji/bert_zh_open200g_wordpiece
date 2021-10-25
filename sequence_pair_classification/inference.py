import sys
import torch
import argparse
sys.path.append('..')
from bert import BERTLM
from main import myModel
import numpy as np
from google_bert import BasicTokenizer

def extract_parameters(ckpt_path):
    model_ckpt = torch.load(ckpt_path)
    bert_args = model_ckpt['bert_args']
    model_args = model_ckpt['args']
    bert_vocab = model_ckpt['bert_vocab']
    model_parameters = model_ckpt['model']
    return bert_args, model_args, bert_vocab, model_parameters

def init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx = 'none'):
    bert_model = BERTLM(gpu_id, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, \
            bert_args.dropout, bert_args.layers, approx)
    return bert_model

def init_sequence_classification_model(empty_bert_model, args, bert_args, gpu_id, bert_vocab, model_parameters):
    number_class = args.number_class
    embedding_size = bert_args.embed_dim
    batch_size = args.batch_size
    dropout = args.dropout
    device = gpu_id
    vocab = bert_vocab
    seq_tagging_model = myModel(empty_bert_model, number_class, embedding_size, batch_size, dropout, 
        device, vocab)
    seq_tagging_model.load_state_dict(model_parameters)
    return seq_tagging_model

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_data',type=str)
    parser.add_argument('--out_path',type=str)
    parser.add_argument('--gpu_id',type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    ckpt_path = args.ckpt_path
    test_data = args.test_data
    out_path = args.out_path
    gpu_id = args.gpu_id

    bert_args, model_args, bert_vocab, model_parameters = extract_parameters(ckpt_path)
    empty_bert_model = init_empty_bert_model(bert_args, bert_vocab, gpu_id, approx = 'none')
    seq_classification_model = init_sequence_classification_model(empty_bert_model, model_args, 
            bert_args, gpu_id, bert_vocab, model_parameters)
    seq_classification_model.cuda(gpu_id)

    tokenizer = BasicTokenizer()

    seq_classification_model.eval()
    with torch.no_grad():
        with open(out_path, 'w', encoding = 'utf8') as o:
            with open(test_data, 'r', encoding = 'utf8') as i:
                lines = i.readlines()
                for l in lines:
                    content_list = l.strip('\n').split('\t')
                    text = content_list[0]
                    text_tokenized_list = tokenizer.tokenize(text)
                    if len(text_tokenized_list) > args.max_len:
                        text_tokenized_list = text_tokenized_list[:args.max_len]
                    pred_output = seq_classification_model([text_tokenized_list], fine_tune = False).cpu().numpy()
                    pred_probability = pred_output[0] 
                    pred_label = np.argmax(pred_probability)
                    out_line = text + '\t' + str(pred_label)
                    o.writelines(out_line + '\n')
    print("done.")
