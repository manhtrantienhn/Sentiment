import random
import os

import numpy as np
import torch
import nltk
from torch.utils.data import TensorDataset, DataLoader
nltk.download('punkt')
import argparse
import os
from transformers import AutoTokenizer
from utils.model import *
from utils.data_loader import pad_and_truncate_seqs, create_vocab, read_data, text_to_seq
from train import initialize_model, validate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'using seed: %d' %(seed))

def create_data_loader(train_dataset, test_dataset, batch_size, origin_max_len, mask_max_len):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print('reading datasets...')
    train_data, test_data = read_data(train_dataset, test_dataset)

    print('creating vocab...')
    origin_w2idx, _, mask_w2idx, _, sos_idx, eos_idx, pad_idx, unk_idx = create_vocab(train_data)

    print('convert text to sequence...')
    origin_test = text_to_seq(test_data['origin'], origin_w2idx, unk_idx)
    restr_test = text_to_seq(test_data['resconstruct'], origin_w2idx, unk_idx)
    mask_test = text_to_seq(test_data['mask'], mask_w2idx, unk_idx)

    print('padding and truncating seq...')
    origin_test_pad = pad_and_truncate_seqs(origin_test, origin_max_len, pad_idx, sos_idx, eos_idx)
    restr_test_pad = pad_and_truncate_seqs(restr_test, origin_max_len, pad_idx, sos_idx, eos_idx)
    mask_test_pad = pad_and_truncate_seqs(mask_test, mask_max_len, pad_idx, sos_idx, eos_idx)
    
    origin_bert_token_test = tokenizer.batch_encode_plus(test_data['origin'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    restr_bert_token_test = tokenizer.batch_encode_plus(test_data['resconstruct'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')
    mask_bert_token_test = tokenizer.batch_encode_plus(test_data['mask'], padding=True, truncation=True, max_length=mask_max_len, return_tensors='pt')

    test_tensor = TensorDataset(torch.tensor(origin_test_pad, dtype=torch.long), origin_bert_token_test['input_ids'], origin_bert_token_test['attention_mask'],
                               torch.tensor(restr_test_pad, dtype=torch.long), restr_bert_token_test['input_ids'], restr_bert_token_test['attention_mask'],
                               torch.tensor(mask_test_pad, dtype=torch.long), mask_bert_token_test['input_ids'], mask_bert_token_test['attention_mask'])
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return test_loader, origin_w2idx, mask_w2idx

def main():
    parser = argparse.ArgumentParser(description='Sentiment Model')

    parser.add_argument('--train_dataset', type=str, default='./train.json', help='path to train dataset')
    parser.add_argument('--val_dataset', type=str, default='./val.json', help='path to val dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best checkpoint')
    args = parser.parse_args()

    TRAIN_PATH = args.train_dataset
    VAL_PATH = args.val_dataset
    CHECK_POINT = args.checkpoint
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    config_file = os.path.join('/'.join(CHECK_POINT.split('/')[:-1]), 'config.pth')
    print('load config from: ', config_file)
    config = torch.load(config_file)
    BATCH_SIZE = config['batch_size']
    OR_MAX_LENGTH = config['origin_max_len']
    MASK_MAX_LENGTH = config['mask_max_len']
    HIDDEN_SIZE = config['hidden_size']
    EMBEDD_DIM = config['embedding_dim']
    NUM_ASPECT = config['num_aspect']
    IGNORE_INDEX = config['ignore_index']
    LR = config['lr']
    LEN_TRAIN_ITER = config['len_train_iter']
    EPOCHS = config['epochs']


    print('reading dataset...')
    test_loader, origin_w2idx, mask_w2idx = create_data_loader(train_dataset=TRAIN_PATH,
                                                                test_dataset=VAL_PATH,
                                                                batch_size=BATCH_SIZE,
                                                                origin_max_len=OR_MAX_LENGTH,
                                                                mask_max_len=MASK_MAX_LENGTH)

    print('initializing model...')
    model, criterion, _, _ = initialize_model(origin_vocab=len(origin_w2idx),
                                                          restr_vocab=len(origin_w2idx),
                                                          mask_vocab=len(mask_w2idx),
                                                          hidden_size=HIDDEN_SIZE,
                                                          embedding_dim=EMBEDD_DIM,
                                                          len_train_iter=LEN_TRAIN_ITER,
                                                          num_aspect=NUM_ASPECT, device=DEVICE,
                                                          ignore_index=IGNORE_INDEX,
                                                          epochs=EPOCHS, lr=LR)

    print('Testing model')

    print('loading checkpoint from: ', CHECK_POINT)
    checkpoint = torch.load(CHECK_POINT)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Val loss: {}'.format(checkpoint['val_loss']))

    # validate model
    print('Testing model...')
    test_loss = validate(model, criterion, test_loader, DEVICE)
    print("Test loss: {}".format(test_loss))

if __name__ == '__main__':
    main()