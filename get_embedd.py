import random
import os

import numpy as np
import torch
import json
import nltk
from torch.utils.data import TensorDataset, DataLoader
nltk.download('punkt')
import argparse
import os
from transformers import AutoTokenizer
from utils.model import *
from utils.data_loader import pad_and_truncate_seqs, create_vocab, read_data, text_to_seq
from train import initialize_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'using seed: %d' %(seed))

def create_data_loader(train_dataset, test_dataset, batch_size, origin_max_len, test_max_len=50):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print('reading datasets...')
    train_data, test_data = read_data(train_dataset, test_dataset)

    print('creating vocab...')
    origin_w2idx, origin_idx2w, mask_w2idx, _, sos_idx, eos_idx, pad_idx, unk_idx = create_vocab(train_data)

    print('convert text to sequence...')
    origin_train = text_to_seq(train_data['origin'], origin_w2idx, unk_idx)
    test_seqs = text_to_seq(test_data['text'], origin_w2idx, unk_idx)

    print('padding and truncating seq...')
    train_pads = pad_and_truncate_seqs(origin_train, origin_max_len, pad_idx, sos_idx, eos_idx)
    test_pads = pad_and_truncate_seqs(test_seqs, test_max_len, pad_idx, sos_idx, eos_idx)
    
    test_token = tokenizer.batch_encode_plus(test_data['text'], padding=True, truncation=True, max_length=test_max_len, return_tensors='pt')
    train_token = tokenizer.batch_encode_plus(train_data['origin'], padding=True, truncation=True, max_length=origin_max_len, return_tensors='pt')

    train_tensor = TensorDataset(torch.tensor(train_pads, dtype=torch.long), train_token['input_ids'], train_token['attention_mask'])
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    test_tensor = TensorDataset(torch.tensor(test_pads, dtype=torch.long), test_token['input_ids'], test_token['attention_mask'])
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, origin_w2idx, origin_idx2w, mask_w2idx

def main():
    parser = argparse.ArgumentParser(description='Sentiment Model')

    parser.add_argument('--train_dataset', type=str, default='./train.json', help='path to train dataset')
    parser.add_argument('--test_dataset', type=str, default='./test.json', help='path to val dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best checkpoint')
    args = parser.parse_args()

    TRAIN_PATH = args.train_dataset
    TEST_PATH = args.test_dataset
    CHECK_POINT = args.checkpoint
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    checkpoint_path = '/'.join(CHECK_POINT.split('/')[:-1])

    config_file = os.path.join(checkpoint_path, 'config.pth')
    print('load config from: ', config_file)
    config = torch.load(config_file)
    BATCH_SIZE = config['batch_size']
    OR_MAX_LENGTH = config['origin_max_len']
    HIDDEN_SIZE = config['hidden_size']
    EMBEDD_DIM = config['embedding_dim']
    NUM_ASPECT = config['num_aspect']
    IGNORE_INDEX = config['ignore_index']
    LR = config['lr']
    LEN_TRAIN_ITER = config['len_train_iter']
    EPOCHS = config['epochs']


    print('reading dataset...')
    train_loader, test_loader, origin_w2idx, origin_idx2w, mask_w2idx = create_data_loader(train_dataset=TRAIN_PATH, test_dataset=TEST_PATH, batch_size=BATCH_SIZE,
                                                                            origin_max_len=OR_MAX_LENGTH, test_max_len=50)

    w2idx_path = os.path.join(checkpoint_path, 'w2idx.json')                                                            
    print('saving corpus w2idx to: ', w2idx_path)
    with open(w2idx_path, 'w+') as f:
        json.dump(origin_w2idx, f, indent=4)

    idx2w_path = os.path.join(checkpoint_path, 'w2idx.json')
    print('saving corpus idx2w to: ', idx2w_path)
    with open(idx2w_path, 'w+') as f:
        json.dump(origin_idx2w, f, indent=4)

    print('initializing model...')
    model, _, _, _ = initialize_model(origin_vocab=len(origin_w2idx),
                                    restr_vocab=len(origin_w2idx),
                                    mask_vocab=len(mask_w2idx),
                                    hidden_size=HIDDEN_SIZE,
                                    embedding_dim=EMBEDD_DIM,
                                    len_train_iter=LEN_TRAIN_ITER,
                                    num_aspect=NUM_ASPECT, device=DEVICE,
                                    ignore_index=IGNORE_INDEX,
                                    epochs=EPOCHS, lr=LR)


    print('loading checkpoint from: ', CHECK_POINT)
    checkpoint = torch.load(CHECK_POINT)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder = model.encoder

    print('compute average word embedding of each token in vocab...')
    w2e = {i:[w, 1] for i, w in origin_idx2w.items()}
    with torch.no_grad():
        for _, batch in enumerate(train_loader):
            x, input_ids, attention_mask = tuple(t.to(DEVICE) for t in batch)

            (output1, _), (output2, _) = encoder(input_ids, attention_mask, x)
            output = torch.cat((output1, output2), dim=2) # [batch_size, seq_len, 1600]
            for row in x:
                for i in row:
                    if i in w2e:
                        w2e[i][0] += output[row, i].detach().cpu().numpy()
                        w2e[i][1] += 1
    
    w2embedd = {i:list(w[0]/w[1]) for i, w in w2e}
    print('saving word to embedding to: ', os.path.join(checkpoint_path, 'w2e.json'))
    with open(os.path.join(checkpoint_path, 'w2e.json'), 'w+') as f:
        json.dump(w2embedd, f)
    # torch.save(w2e, os.path.join(checkpoint_path, 'w2e.pt'))

    sentence_embs = []
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            x, input_ids, attention_mask = tuple(t.to(DEVICE) for t in batch)

            (output1, _), (output2, _) = encoder(input_ids, attention_mask, x)
            sentence_emb = torch.cat((output1, output2), dim=2)[:, -1].detach().cpu().numpy() # [batch_size, seq_len, 1600]

            sentence_embs.extend[sentence_emb]
    
    # sentence_embs = {i:emb for i, emb in enumerate(sentence_embs)}
    print('saving sentences embedding to: ', os.path.join(checkpoint_path, 'sent_embedd.json'))
    with open(os.path.join(checkpoint_path, 'sent_embedd.json'), 'w+') as f:
        json.dump({'embedd':sentence_embs}, f)

    
    print('saving matrix T to: ', os.path.join(checkpoint_path, 'T.txt'))
    np.savetxt(os.path.join(checkpoint_path, 'T.txt'), model.T.weight.detach().numpy())

    # print('saving weight of matrix T to: ', os.path.join(checkpoint_path, 'T.pt'))
    # torch.save({'T': model.T}, os.path.join(checkpoint_path, 'T.pt'))

    # print('Val loss: {}'.format(checkpoint['val_loss']))

    # validate model
    # print('Testing model...')
    # test_loss = validate(model, criterion, test_loader, DEVICE)
    # print("Test loss: {}".format(test_loss))

if __name__ == '__main__':
    main()