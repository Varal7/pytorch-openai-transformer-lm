from pathlib import Path
import numpy as np
import sklearn.model_selection
import pandas as pd

import argparse
import os
import random


import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from datasets import imdb
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                   ResultLogger, make_path)
from loss import ClassificationLossCompute


encoder_path = 'model/encoder_bpe_40000.json'
bpe_path = 'model/vocab_40000.bpe'
data_dir = '../datasets/aclImdb/'
n_ctx = 512
n_batch = 2
log_dir = 'log'
desc = 'imdb'
n_iter = 3
submit = False
save_dir = 'save'

# encoder_path = args.encoder_path
# bpe_path = args.bpe_path
# n_batch = args.n_batch

# submit = args.submit
# dataset = args.dataset
# n_ctx = args.n_ctx
# save_dir = args.save_dir
# desc = args.desc # data_dir = args.data_dir
# log_dir = args.log_dir
# submission_dir = args.submission_dir
# iter = args.n_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("device", device, "n_gpu", n_gpu)

# logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)))

text_encoder = TextEncoder(encoder_path, bpe_path)
encoder = text_encoder.encoder
n_vocab = len(text_encoder.encoder)


print("Encoding dataset...")

((trX, trY), (vaX, vaY), _) = encode_dataset(*imdb(data_dir, n_train=100, n_valid=1000),
                                        encoder=text_encoder)


encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_']
n_special = 3
max_len = n_ctx - 2
vocab = n_vocab + n_special + n_ctx

def transform_imdb(X):
    n_batch = len(X)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, x in enumerate(X):
        xr = [start] + x[:max_len] + [clf_token]
        lr = len(xr)
        xmb[i, :lr, 0] = xr
        mmb[i, :lr] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb


trX, trM = transform_imdb(trX)
vaX, vaM = transform_imdb(vaX)

n_train = len(trY)
n_valid = len(vaY)


n_batch_train = n_batch * max(n_gpu, 1)
n_updates_total = (n_train // n_batch_train) * n_iter

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})

args = dotdict({
    'lr': 6.25e-5,
    'lr_schedule': 'warmup_linear',
    'lr_warmup': 0.002,
    'b1': 0.9,
    'b2': 0.999,
    'e': 1e-8,
    'l2': 0.01,
    'vector_l2': False,
    'max_grad_norm': 1,
    'lm_coef': 0.5,
})

dh_model = DoubleHeadModel(DEFAULT_CONFIG, clf_token, ('classification', 2), vocab, n_ctx)

criterion = nn.CrossEntropyLoss(reduce=False)

model_opt = OpenAIAdam(dh_model.parameters(),
                        lr=args.lr,
                        schedule=args.lr_schedule,
                        warmup=args.lr_warmup,
                        t_total=n_updates_total,
                        b1=args.b1,
                        b2=args.b2,
                        e=args.e,
                        l2=args.l2,
                        vector_l2=args.vector_l2,
                        max_grad_norm=args.max_grad_norm)

compute_loss_fct = ClassificationLossCompute(criterion,
                                                criterion,
                                                args.lm_coef,
                                                model_opt)

load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

dh_model.to(device)
dh_model = nn.DataParallel(dh_model)

def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)

def iter_apply(Xs, Ms, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    if submit:
        score = va_acc
        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, desc, 'best_params')
            torch.save(dh_model.state_dict(), make_path(path))



n_updates = 0
n_epochs = 0
trYt = trY


if submit:
    path = os.path.join(save_dir, desc, 'best_params')
    torch.save(dh_model.state_dict(), make_path(path))
best_score = 0

for i in range(n_iter):
    print("running epoch", i)
    run_epoch()
    n_epochs += 1
    log(save_dir, desc)

if submit:
    path = os.path.join(save_dir, desc, 'best_params')
    dh_model.load_state_dict(torch.load(path))
    predict(dataset, args.submission_dir)
    if args.analysis:
        rocstories_analysis(data_dir, os.path.join(args.submission_dir, 'ROCStories.tsv'),
                            os.path.join(log_dir, 'rocstories.jsonl'))
