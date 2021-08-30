# Sentiment analysis with unsupervised model

## Directory
    ├── checkpoint/ 
    ├── data/
    │   ├── fasttext.vec
    │   ├── fasttext_mask.vec
    │   ├── stm.vec
    │   ├── stm_mask.vec
    │   ├── train.json
    │   ├── val.json
    │   ├── test.json
    ├── utils/
    │   ├── __init__.py
    │   ├── dataloader.py
    │   ├── model.py
    ├── train.py
    ├── test.py
    └── get_embedd.py


## Train

For training, you can run commands like this:

```shell
python3 train.py --train_dataset ./data/train.json --val_dataset ./data/val.json --batch_size 256 --epochs 20 --hidden_size 400 --checkpoint ./checkpoint/ --lr 5e-3 --num_aspect 14 --patience 5 --delta 1e-6 --fasttext_vec ./data/fasttext.vec --fasttext_mask_vec ./data/fasttext_mask.vec --stm_vec ./data/stm.vec --stm_mask_vec ./data/stm_mask.vec
```

## Test

For evaluation, the command may like this:

```shell
python3 test.py --train_dataset ./data/train.json --test_dataset ./data/test.json --checkpoint ./checkpoint/path_to_the_best_checkpoint
```