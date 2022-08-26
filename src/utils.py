import numpy as np
import os
import pandas as pd
from typing import List, Union, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import scale

def plot_energy_distribution(df : pd.DataFrame, col : Literal['Reorg_g','Reorg_ex'] = 'Reorg_g'):
    sns.distplot(df[col], hist = True, color = 'purple', axlabel = col)
    plt.title("train data : {} distribution".format(col))
    plt.ylabel("# of records(relative)")

    if os.path.exists("./result"):
        plt.savefig("./result/{}_distribution.png".format(col))
    else:
        os.mkdir("./result")
        plt.savefig("./result/{}_distribution.png".format(col))

def plot_training_curve(save_dir : str, train_loss : Union[List, np.array, np.ndarray], valid_loss : Union[List, np.array, np.ndarray]):
    x_epoch = range(1, len(train_loss) + 1)
    plt.figure(figsize = (9,6))
    plt.plot(x_epoch, train_loss, 'ro-', label = 'train loss')
    plt.plot(x_epoch, valid_loss, 'bo-', label = 'valid loss')
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel('loss')
    plt.savefig(save_dir)

# extract alphabet and index dictionay from dataframe
def extract_alphabet_dict(df : pd.DataFrame):
    corpus = df.copy()['SMILES']
    corpus['sequence'] = corpus.map(list)

    alphabet_list = []

    from tqdm.auto import tqdm
    for corpus in tqdm(corpus['sequence'].values):
        corpus = np.unique(corpus).tolist()
        alphabet_list.extend(corpus)

    alphabet_combination = np.unique(alphabet_list)

    print("total alphabet : ", len(alphabet_combination))

    # dictionary
    char2idx = {}
    idx2char = {}

    for idx, char in enumerate(alphabet_combination):
        idx2char[idx] = char
        char2idx[char] = idx

    idx2char[idx+1] = '<PAD>'
    char2idx['<PAD>'] = idx+1

    return char2idx, idx2char