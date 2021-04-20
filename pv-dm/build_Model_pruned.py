# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:25:32 2019

"""

import data_processing as dp
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
from util import timeSince
import numpy as np
import os
import pandas as pd
import sims

from constants import DATA_PATH_DIR, DATA_PATH_CSV, MODEL_FILE,EMBEDDING_DIM



def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()[2:]
    #print(s)
    des = s[s.index('\n')+3:-2]
    t = s[:s.index('\n')-1]
    return t,des

def splitfile(fromPath, toPath):
    skipped_labels = False
    line_number = 1
    with open(fromPath, 'r', encoding='utf-8') as infile:
        for text in infile:
            # print("Reading line " + str(line_number))
            line_number += 1
            if skipped_labels:
                ID = ""
                for char in text:
                    if char not in '"[],/':
                        ID += char
                    if char in ",/":
                        break
                # if ID[5:15] == '0000789019' or ID[5:15] == '0000886982' or ID[5:15] == '0000019617':
                outfile = open(toPath + "/" + ID, 'w', encoding='utf-8')
                outfile.write(text)
                outfile.close()
            else:
                skipped_labels = True

def split_by_year(fNames):
    sections = []
    vlist = sorted(fNames, key=lambda s:s[16:20])
    sect = []
    year = vlist[0][16:20]
    for f in vlist:
        if f[16:20] == year:
            sect = sect + [f]
        else:
            sections.append(sect)
            sect = [f]
            year = f[16:20]
    return sections

def train_vectors(data_path, model_file,nouns=True):
    """
    Trains a doc2vec model from preprocessed text, then saves it to a file.
    
    """
    start = time.time()
    print("getting data")
    ids,sentences_ls=dp.get_paragraphs(data_path)
    print("got data")
    if nouns:
        tagged_data = [TaggedDocument(words=dp.preprocess_str_hp(_d).split(), tags=[str(i)]) for i, _d in sentences_ls]
    else:
        tagged_data = [TaggedDocument(words=dp.preprocess(_d).split(), tags=[str(i)]) for i, _d in sentences_ls]
    print("made ", len(sentences_ls), " sentences")
    print("sample: ", tagged_data[0])
    embedding=build_model(tagged_data)
    print("made model in ",timeSince(start))
    embedding.save(model_file)
    return ids,embedding

def build_model(tagged_data,dim=EMBEDDING_DIM,epochs=20,lr=0.025,dm_model=1):
    model = Doc2Vec(vector_size=dim,
                alpha=lr, 
                min_alpha=lr,
                min_count=1,
                dm =1,
                epochs=1
                )
    print(model.epochs)
    model.build_vocab(tagged_data)
    for epoch in range(epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    return model


def make_sims(vecName, year, prefix, diag=0.0):
    outName ='10k2v_sims_' + year + '.csv'
    sims.run_sims(vecName, outName,'all10k')

def make_vec_df(model,path, prefix):
    num_files = len(os.listdir(path))
    years = split_by_year(sorted(os.listdir(path)))
    i=1
    start = time.time()
    for year in years:
        data = []
        firms = []
        print(i/num_files, 'in ', str(timeSince(start)))
        for filename in year:
            file = open(path+"/"+filename,'r', encoding="utf8")
            #print(file)
            title,des=t_n_d(file)
            try:
                des_ls = dp.preprocess_str_hp(des).split()
                vec = list(model.infer_vector(des_ls))
                data.append(vec)
                firms.append(title)
            except:
                pass
                # print("missed ", filename)
            i+=1
        data = np.array(data).T.tolist()
        df = pd.DataFrame(data, columns=firms)
        print('made vecs')
        #print(df[:5])
        outname = '10k2v_vectors_' + year[0][16:20] + '.csv'
        df.to_csv(outname)
        make_sims(outname, year[0][16:20], prefix, 0.0)
        print('made sims')


if __name__ == "__main__":
    print("training d2v")
    #ids,model=train_vectors(DATA_PATH_CSV, MODEL_FILE)
    model = Doc2Vec.load(MODEL_FILE)
    print('loaded model')
    make_vec_df(model,DATA_PATH_DIR,"",)
    print('done')