import random
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import nltk
nltk.download('punkt')
import argparse
import os

from utils.model import *
from utils.data_loader import create_data_loader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'using seed: %d' %(seed))

def initialize_model(top_s_r_embedding, bot_s_r_embedding, top_m_embedding, bot_m_embedding, hidden_size, len_train_iter, num_aspect, device, ignore_index, epochs=20, lr=3e-5):

    encoder = Encoder(hidden_size=hidden_size,
                      device=device,
                      weight_top_embedding=top_s_r_embedding,
                      weight_bot_embedding=bot_s_r_embedding)

    decoder1 = Decoder(hidden_size=hidden_size,
                       weight_top_embedding=top_s_r_embedding,
                       weight_bot_embedding=bot_s_r_embedding,
                       encoder_output_dim=(encoder.hidden_size*encoder.D))
    
    decoder2 = Decoder(hidden_size=hidden_size,
                       weight_top_embedding=top_m_embedding,
                       weight_bot_embedding=bot_m_embedding,
                       encoder_output_dim=(encoder.hidden_size*encoder.D))
    
    model = SentimentModel(encoder=encoder, decoder1=decoder1, decoder2=decoder2, num_aspect=num_aspect).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    n_steps = len_train_iter * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=100)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    return model, criterion, optimizer, scheduler

def step(model, criterion, optimizer, scheduler, batch, device):
    origin, res, mask = batch

    optimizer.zero_grad()
    
    output_reverts, output_masks = model(top_source=origin, bot_source=origin, top_res=res, top_mask=mask, teacher_forcing_ratio=0.5)

    output_reverts, output_masks = output_reverts.transpose(1, 0), output_masks.transpose(1, 0)

    output_reverts_flatten = output_reverts[:, 1:].reshape(-1, output_reverts.shape[-1]).to(device)
    output_masks_flatten = output_masks[:, 1:].reshape(-1, output_masks.shape[-1]).to(device)

    target_reverts_flatten = res[:, 1:].reshape(-1)
    target_masks_flatten = mask[:, 1:].reshape(-1)

    loss = criterion(output_reverts_flatten, target_reverts_flatten) + criterion(output_masks_flatten, target_masks_flatten)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    scheduler.step()

    return loss.item()

def validate(model, criterion, val_iterator, device, teacher_forcing_ratio=0.0):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in val_iterator:
            # revert_preds, revert_targets, mask_preds, mask_targets = list(), list(), list(), list()
            origin, res, mask = batch

            output_reverts, output_masks = model(top_source=origin, bot_source=origin, top_res=res, top_mask=mask, teacher_forcing_ratio=teacher_forcing_ratio)

            output_reverts, output_masks = output_reverts.transpose(1, 0), output_masks.transpose(1, 0)

            output_reverts_flatten = output_reverts[:, 1:].reshape(-1, output_reverts.shape[-1]).to(device)
            output_masks_flatten = output_masks[:, 1:].reshape(-1, output_masks.shape[-1]).to(device)

            target_reverts_flatten = res[:, 1:].reshape(-1)
            target_masks_flatten = mask[:, 1:].reshape(-1)
            loss = criterion(output_reverts_flatten, target_reverts_flatten) + criterion(output_masks_flatten, target_masks_flatten)

            running_loss += loss.item()
            # revert_preds.extend(output_reverts_flatten.argmax(1).detach().cpu())
            # mask_preds.extend(output_masks_flatten.argmax(1).detach().cpu())

            # revert_targets.extend(target_reverts_flatten)
            # mask_targets.extend(target_masks_flatten)

        val_loss = running_loss/(len(val_iterator))
        # val_acc = (accuracy_score(revert_preds, revert_targets) + accuracy_score(mask_preds, mask_targets))/2
    return val_loss


def train(model, criterion, optimizer, scheduler, train_iterator, val_iterator, checkpoint, epochs, device, patience, delta):
    
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_iterator):
            loss = step(model, criterion, optimizer, scheduler, batch, device)

            running_loss += loss

            if (i+1) % 50 == 0 or i == 0:
                print("Epoch: {}/{} - iter: {}/{} - train_loss: {}".format(epoch+1, epochs, i + 1, len(train_iterator), running_loss/(i + 1)))
        else:
            train_loss = running_loss/len(train_iterator)
            print("Epoch: {}/{} - iter: {}/{} - train_loss: {}\n".format(epoch+1, epochs, i + 1, len(train_iterator), train_loss))

            print('Evaluating...')
            val_loss = validate(model, criterion, val_iterator, device)
            print("    Val loss: {}\n".format(val_loss))

            torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss' : val_loss
            }, os.path.join(checkpoint, 'cp'+str(epoch+1)+'pt'))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                train_losses, val_losses = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)
                np.savetxt(os.path.join(checkpoint, 'log_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')
                print('Early stoppping.')
                break

    train_losses, val_losses = np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)
    np.savetxt(os.path.join(checkpoint, 'log_loss.txt'), np.hstack((train_losses, val_losses)), delimiter='#')

def main():
    parser = argparse.ArgumentParser(description='Sentiment Model')

    parser.add_argument('--train_dataset', type=str, default='./train.json', help='path to train dataset')
    parser.add_argument('--val_dataset', type=str, default='./val.json', help='path to val dataset')
    parser.add_argument('--batch_size', type=int, default= 256, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number epochs')
    parser.add_argument('--hidden_size', type=int, default=400, help='hidden size of rnns layer')
    parser.add_argument('--or_max_len', type=int, default= 39, help='max sequence length of origin and resconstruct data')
    parser.add_argument('--mask_max_len', type=int, default=15, help='max sequence length of mask data')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to check point directory')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--num_aspect', type=int, default=14, help='number of aspect in dataset')
    parser.add_argument('--patience', type=int, default=5, help='patience using in early stopping')
    parser.add_argument('--delta', type=float, default=1e-6, help='delta using in early stopping')
    parser.add_argument('--fasttext_vec', type=str, default='./data/fasttext.vec')
    parser.add_argument('--fasttext_mask_vec', type=str, default='./data/fasttext_mask.vec')
    parser.add_argument('--stm_vec', type=str, default='./data/stm.vec')
    parser.add_argument('--stm_mask_vec', type=str, default='./data/stm_mask.vec')

    args = parser.parse_args()

    TRAIN_PATH = args.train_dataset
    VAL_PATH = args.val_dataset
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    HIDDEN_SIZE = args.hidden_size
    OR_MAX_LENGTH = args.or_max_len
    MASK_MAX_LENGTH = args.mask_max_len
    CHECK_POINT = args.checkpoint
    LR = args.lr
    NUM_ASPECT = args.num_aspect
    PATIENCE = args.patience
    DELTA = args.delta
    FASTTEXT_VEC = args.fasttext_vec
    FASTTEXT_MASK_VEC = args.fasttext_mask_vec
    STM_VEC = args.stm_vec
    STM_MASK_VEC = args.stm_mask_vec
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_seed()
    print('reading dataset...')
    train_loader, val_loader, w2idx, _, fwe, fmwe, swe, smwe = create_data_loader(train_dataset=TRAIN_PATH, valid_dataset=VAL_PATH,
                                                                                  batch_size=BATCH_SIZE, device=DEVICE, origin_max_len=OR_MAX_LENGTH, 
                                                                                  mask_max_len=MASK_MAX_LENGTH, fasttext_vec=FASTTEXT_VEC,
                                                                                  fasttext_mask_vec=FASTTEXT_MASK_VEC, stm_vec=STM_VEC,
                                                                                  stm_mask_vec=STM_MASK_VEC)
    # fwe: fasttext word embedding, fmwe: fasttext mask word embedding
    # swe: sentiment word embedding, smwe: sentiment mask word embedding

    print('initializing model...')
    model, criterion, optimizer, scheduler = initialize_model(top_s_r_embedding=fwe, bot_s_r_embedding=swe, 
                                                              top_m_embedding=fmwe, bot_m_embedding=smwe, 
                                                              hidden_size=HIDDEN_SIZE, len_train_iter=len(train_loader), 
                                                              num_aspect=NUM_ASPECT, device=DEVICE, 
                                                              ignore_index=w2idx['<pad>'], 
                                                              epochs=EPOCHS, lr=LR)

    print('saving model config to: ', os.path.join(CHECK_POINT, 'config.pt'))
    torch.save({
        'top_s_r_embedding': fwe, 'bot_s_r_embedding': swe, 'top_m_embedding': fmwe, 'bot_m_embedding': smwe,
        'batch_size': BATCH_SIZE, 'origin_max_len': OR_MAX_LENGTH, 'mask_max_len': MASK_MAX_LENGTH,
        'epoch': EPOCHS, 'hidden_size':HIDDEN_SIZE, 'len_train_iter': len(train_loader),
        'num_aspect':NUM_ASPECT, 'ignore_index': w2idx['<pad>'], 'lr':LR
        }, os.path.join(CHECK_POINT, 'config.pt'))

    print('training model...')
    train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_iterator=train_loader,
          val_iterator=val_loader, checkpoint=CHECK_POINT, epochs=EPOCHS, device=DEVICE, patience=PATIENCE, delta=DELTA)

if __name__ == '__main__':
    main()