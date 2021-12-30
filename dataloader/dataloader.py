"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from argparse import Namespace
import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class TextClustering(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

class AugmentPairSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'label': self.train_y[idx]}


def augment_loader(args:Namespace):
    if args.dataset == "searchsnippets":
        train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
        train_text = train_data['text'].fillna('.').values
        train_text1 = train_data['text1'].fillna('.').values
        train_text2 = train_data['text2'].fillna('.').values
        train_label = train_data['label'].astype(int).values

    elif args.dataset == "bili":
        DATALEN = args.datalen
        data_path = args.data_path
        aug_path =  data_path+args.aug_path
        #sub_areas = ['science','social_science','humanity_history','business','campus','career','design','skill']
        #sub_areas = ['douga','music','dance','game','knowledge','tech','sports','car','life','food','animal','fashion','information','ent']
        sub_areas = ['music','tech','sports','car','food','animal','fashion','information','ent']
        train_text = []
        train_text1 = []
        train_text2 = []
        train_label = []
        for idx,sub_area in enumerate(sub_areas):
            with open(data_path+sub_area+'.txt',encoding='utf-8') as dataFile:
                dataList = dataFile.read().split('\n')
                if DATALEN:
                    dataList = dataList[:DATALEN]
                dataList = [i[13:] for i in dataList]
                train_text.extend(dataList)
            with open(aug_path+sub_area+'1.txt',encoding='utf-8') as dataFile:
                dataList = dataFile.read().split('\n')
                if DATALEN:
                    dataList = dataList[:DATALEN]
                train_text1.extend(dataList)
            with open(aug_path+sub_area+'2.txt',encoding='utf-8') as dataFile:
                dataList = dataFile.read().split('\n')
                if DATALEN:
                    dataList = dataList[:DATALEN]
                train_text2.extend(dataList)
                for i in range(len(dataList)):
                    train_label.append(idx)

            assert len(train_text) == len(train_text1) == len(train_text2) == len(train_label)

            

    else:
        DATALEN = args.datalen
        with open('data/stackoverflow/title_StackOverflow.txt',encoding='utf-8') as dataFile:
            train_text = dataFile.read().split('\n')[:DATALEN]
        with open('data/stackoverflow/text1.txt',encoding='utf-8') as dataFile:
            train_text1 = dataFile.read().split('\n')[:DATALEN]
        with open('data/stackoverflow/text2.txt',encoding='utf-8') as dataFile:
            train_text2 = dataFile.read().split('\n')[:DATALEN]
        with open('data/stackoverflow/label_StackOverflow.txt',encoding='utf-8') as dataFile:
            train_label = [int(i)-1 for i in dataFile.read().split('\n')[:DATALEN]]


    print(max(train_label), min(train_label))
    print(len(train_text) , len(train_text1) , len(train_text2) , len(train_label))
    train_dataset = AugmentPairSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return train_loader

def train_unshuffle_loader(args):
    if args.dataset == "searchsnippets":
        train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
        train_text = train_data['text'].fillna('.').values
        train_label = train_data['label'].astype(int).values

    elif args.dataset == "bili":
        DATALEN = args.datalen
        data_path = args.data_path
        #sub_areas = ['science','social_science','humanity_history','business','campus','career','design','skill']
        sub_areas = ['music','tech','sports','car','food','animal','fashion','information','ent']
        train_text = []
        train_label = []
        for idx,sub_area in enumerate(sub_areas):
            with open(data_path+sub_area+'.txt',encoding='utf-8') as dataFile:
                dataList = dataFile.read().split('\n')
                if DATALEN:
                    dataList = dataList[:DATALEN]
                train_text.extend(dataList)
                if DATALEN:
                    for i in range(DATALEN):
                        train_label.append(idx)
                else:
                    for i in range(len(dataList)):
                        train_label.append(idx)

            assert len(train_text) == len(train_label)

    else:
        DATALEN = args.datalen
        with open('data/stackoverflow/title_StackOverflow.txt',encoding='utf-8') as dataFile:
            train_text = dataFile.read().split('\n')[:DATALEN]
        with open('data/stackoverflow/label_StackOverflow.txt',encoding='utf-8') as dataFile:
            train_label = [int(i)-1 for i in dataFile.read().split('\n')[:DATALEN]]



    train_dataset = TextClustering(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

