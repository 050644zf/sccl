"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import sys
sys.path.append( './' )

import torch
import argparse
from sentence_transformers import SentenceTransformer
from models.Transformers import SCCLBert
from learners.cluster import ClusterLearner
from dataloader.dataloader import augment_loader
from training import training
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path
from utils.randomness import set_global_random_seed

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL_CLASS = {
    "distil": 'distilbert-base-nli-stsb-mean-tokens', 
    "robertabase": 'roberta-base-nli-stsb-mean-tokens',
    "robertalarge": 'roberta-large-nli-stsb-mean-tokens',
    "msmarco": 'distilroberta-base-msmarco-v2',
    "xlm": "xlm-r-distilroberta-base-paraphrase-v1",
    "bertlarge": 'bert-large-nli-stsb-mean-tokens',
    "bertbase": 'bert-base-nli-stsb-mean-tokens',
    "cn": 'data/distiluse-base-multilingual-cased-v1',
    "cndl": 'distiluse-base-multilingual-cased-v1',
    "cn2": 'data/bert-base-chinese',
    "cn2dl": 'bert-base-chinese',
    "cn3": 'data/distiluse-base-multilingual-cased-v2',
    "cn3dl": 'distiluse-base-multilingual-cased-v2',
    "cn4": 'data/paraphrase-multilingual-MiniLM-L12-v2',
    "cn4dl": 'paraphrase-multilingual-MiniLM-L12-v2'
}

def run(args):
    resPath, tensorboard = setup_path(args)
    args.resPath, args.tensorboard = resPath, tensorboard
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader = augment_loader(args)

    # model
    torch.cuda.set_device(args.gpuid[0])
    #配置 Sentence Transformer
    sbert = SentenceTransformer(MODEL_CLASS[args.bert])
    #获取每个聚类的中心
    cluster_centers = get_kmeans_centers(sbert, train_loader, args.num_classes) 
    model = SCCLBert(sbert, cluster_centers=cluster_centers, alpha=args.alpha)  
    model = model.cuda()

    # optimizer 
    optimizer = torch.optim.Adam([
        {'params':model.sentbert.parameters()}, 
        {'params':model.head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}], lr=args.lr)
    print(optimizer)
    
    # set up the trainer    
    learner = ClusterLearner(model, optimizer, args.temperature, args.base_temperature)
    training(train_loader, learner, args)
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=250, help="")  
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--bert', type=str, default='cn4', help="")
    # Dataset
    parser.add_argument('--dataset', type=str, default='bili', help="")
    parser.add_argument('--datalen', type=int, default=100, help="")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--aug_path', type=str, default='augdata/p0.5/')
    parser.add_argument('--dataname', type=str, default='searchsnippets.csv', help="")
    parser.add_argument('--num_classes', type=int, default=8, help="")
    parser.add_argument('--max_length', type=int, default=32)
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=10)
    # contrastive learning
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.07, help="temperature required by contrastive loss")
    # Clustering
    parser.add_argument('--use_perturbation', action='store_true', help="")
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    #args = parser.parse_args('--result_path ./restest/searchsnippets/ --num_classes 8 --dataset bili --bert cn --alpha 1 --lr 1e-05 --lr_scale 100 --batch_size 10 --temperature 0.5 --base_temperature 0.07 --max_iter 10 --print_freq 250 --seed 0 --gpuid 0 '.split(' '))
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args

if __name__ == '__main__':
    run(get_args(sys.argv[1:]))



    
