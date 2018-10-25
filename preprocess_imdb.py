from pathlib import Path
import numpy as np
import pandas as pd

PATH=Path('../datasets/aclImdb/')

CLASSES = ['neg', 'pos', 'unsup']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')

col_names = ['labels','text']

np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]

df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

df_trn[df_trn['labels']!=2].to_csv(PATH/'train.csv', header=False, index=False)
df_val.to_csv(PATH/'test.csv', header=False, index=False)

(PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)

