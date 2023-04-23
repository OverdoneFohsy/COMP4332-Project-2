import pandas as pd
import numpy as np
import os

def word2id(txt,txt2,txt3):
    #prepare wordvector
    data_path = './data'
    word2id = {}
    
    for tokens in txt:
        if not tokens in word2id:
            word2id[tokens] = len(word2id)
                
    for tokens in txt2:
        if not tokens in word2id:
            word2id[tokens] = len(word2id)
    
    for tokens in txt3:
        if not tokens in word2id:
            word2id[tokens] = len(word2id)
                
    word2id['<pad>'] = len(word2id)
    np.save('./data/word2id', word2id)
    print('word2vec created')
    
    return word2id


def texts_to_id_seq(texts, word2id, padding_length=500):
    records = []
    for tokens in texts:
        record = []
        for t in tokens:
            record.append(word2id.get(t, len(word2id)))
        if len(record) >= padding_length:
            records.append(record[:padding_length])
        else:
            records.append(record + [word2id['<pad>']] * (padding_length - len(record)))
    return records
