#!/usr/bin/env python
# coding: utf-8

# ## Part 1 Build on word-level text to generate a fixed-length vector for each sentence

# In[ ]:


import numpy as np 
np.random.seed(2019)
from numpy import genfromtxt

import random as r#
r.seed(2019)

import sys


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)



from util import d, here

import pandas as pd
from argparse import ArgumentParser

import random, sys, math, gzip
from tqdm import tqdm


torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
from argparse import ArgumentParser
from optparse import OptionParser
import easydict


class Normalize(object):
    
    def normalize(self, X_train, X_val):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)
       
        return (X_train, X_val) 
    
    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
    
        return (X_train, X_val) 


def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



def resample(X,Y,random_state=2019):
    X_reshape = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    rus = RandomOverSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_reshape, Y)
    X_back = np.reshape(X_res, (X_res.shape[0],X.shape[1],X.shape[2]))
    
    return X_back, y_res




def initial_seed_dataset(n_initial, Y,random_state):
    
    np.random.seed(random_state)
    
    df = pd.DataFrame()
    df['label'] = Y

    Samplesize = n_initial  #number of samples that you want       
    initial_samples = df.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

    permutation = [index[1] for index in initial_samples.index.tolist()]
    
    print ('initial random chosen samples', permutation)
    
    return permutation





def compute_entropy(output):
    entropy = -(output[:,0]*np.log2(output[:,0]) + output[:,1]*np.log2(output[:,1]))
    return entropy




def sample_candidates(selection_method,permutation,y_prob,num_candidate,loop):
    
    print('*'*20,'selection method ',selection_method,'*'*20)
    


    if selection_method == 'uncertainty':
        entropy = compute_entropy(np.array(y_prob))
        candidate_index = []
        
        
        for index in entropy.argsort()[::-1]:
            if len(candidate_index) == num_candidate:
                break
            if index not in permutation:
                candidate_index.append(index)

        permutation = permutation+candidate_index
        print(np.array(y_prob)[candidate_index])
        
    if selection_method == 'certainty':
        entropy = compute_entropy(np.array(y_prob))
        candidate_index = []
        
        
        for index in entropy.argsort()[:]:
            if len(candidate_index) == num_candidate:
                break
            if index not in permutation:
                candidate_index.append(index)

        permutation = permutation+candidate_index
        
    if selection_method == 'mostConfident':
        entropy = compute_entropy(np.array(y_prob))
        candidate_index = []
        
        
        for index in np.argsort(np.array(y_prob)[:,1])[::-1]:
            if len(candidate_index) == num_candidate:
                break
            if index not in permutation:
                candidate_index.append(index)

        permutation = permutation+candidate_index
        print(np.array(y_prob)[candidate_index])
        
    print('*'*20,'num of training set ',len(permutation),'*'*20)   
        
    return permutation






# ## Part 3. Active Learning Process



def active_process_svm(arg):
    """
    Active learning procedure for Adaptive active learning model.
    """
    
    if arg.loop%arg.gridsearch_interval==0:
        gridsearch = True
    else:
        gridsearch = False
    
    if arg.initial: 
        permutation = initial_seed_dataset(arg.n_seedsamples,arg.LABEL_emb,arg.initial_random_seed)
       
        X_train = arg.TEXT_emb[permutation]
        Y_train = arg.LABEL_emb[permutation]

        X_val = arg.TEXT_emb
        Y_val = arg.LABEL_emb
        
        bin_count = np.bincount(Y_train)
        unique = np.unique(Y_train)
        print (
        'initial training set size:',
        Y_train.shape[0],
        'unique(labels):',
        unique,
        'label counts:',
        bin_count
        )

        print('Number of training examples ', len(Y_train))
        
    else:
        
        permutation = arg.permutation
       
        
        X_train = arg.TEXT_emb[permutation]
        Y_train = arg.LABEL_emb[permutation]

        X_val = arg.TEXT_emb
        Y_val = arg.LABEL_emb
        
        bin_count = np.bincount(Y_train)
        unique = np.unique(Y_train)
        print (
        'training set size:',
        Y_train.shape[0],
        'unique(labels):',
        unique,
        'label counts:',
        bin_count
        )

        print('Number of training examples ', len(Y_train))


    
    
    normalizer = Normalize()
    X_train, X_val = normalizer.normalize(X_train, X_val) 
        
    
    if gridsearch == True:
       
        print('start gridsearch ...')
        parameters = [
                    {'kernel': ['linear'],
                     'C': [ 0.01, 0.1, 1,10]}]

        cv = StratifiedKFold(n_splits=5,random_state=arg.initial_random_seed)
        svc = SVC(probability=True,random_state=2019,class_weight='balanced',max_iter=10000)
        classifier = GridSearchCV(svc, parameters, cv=cv,scoring='accuracy',n_jobs=8,verbose = 0)
        classifier.fit(X_train, Y_train)
        print('best parameters is ', classifier.best_params_)
        best_params_ = classifier.best_params_
       
        
    else:
           
        best_params_ = arg.best_params_
        kernel = best_params_['kernel']
        C = best_params_['C']
        
        classifier = SVC(probability=True,random_state=2019,class_weight='balanced',C=C,kernel=kernel,max_iter=10000)
        classifier.fit(X_train, Y_train)
        print('best parameters is ', classifier)
        
        
    ## evaluation
        
    y_pred_prob = classifier.predict_proba(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
#     y_pred = classifier.predict(X_val)
    y_true = Y_val
    y_eval_prob_pos = np.array(y_pred_prob)[:,1]
    
    numerators = classifier.decision_function(X_val)
    try:
        w_norm = np.linalg.norm(classifier.best_estimator_.coef_)
    except Exception:
        w_norm = np.linalg.norm(classifier.coef_)

    dist = abs(numerators) / w_norm
   

    y_true_remain = np.delete(np.array(y_true),permutation)
    y_pred_remain = np.delete(np.array(y_pred),permutation)
    y_eval_prob_pos_remain = np.delete(y_eval_prob_pos,permutation)
        
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true_remain, y_eval_prob_pos_remain, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
   
    tn, fp, fn, tp = confusion_matrix(y_true_remain, y_pred_remain, labels=[0,1]).ravel()
    print('TP_H',bin_count[1],' TN_H',bin_count[0], ' TP_M',tp, ' TN_M',tn, ' FP_M', fp, ' FN_M',fn)
    acc = accuracy_score(y_true_remain, y_pred_remain)
    f_score = f1_score(y_true_remain,y_pred_remain,average='micro')
                
    raw_result = {'TP_H':bin_count[1],'TN_H':bin_count[0], 'TP_M':tp, 'TN_M':tn, 'FP_M': fp, 'FN_M':fn}
    raw_metrics = {'ACC':acc,'micro_f_score': f_score, 'AUC':auc,'coverage':float(raw_result['TP_H']/(raw_result['TP_H']+raw_result['TP_M']+raw_result['FN_M']))}
    print('ACC:',acc,'micro_f_score:', f_score, 'AUC:',auc)
    print(classification_report(y_true_remain,y_pred_remain))
    
    return raw_result,raw_metrics, y_pred_prob, y_pred,permutation, best_params_

















# Run the main function


if __name__ == "__main__":
    
 


    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -l loop -s selection_strategy -g gridsearch_interval  -c num_candidate -i initial_num_per_class -e text_representation')

    
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by token-level Roberta", default = 'animal_by_product')
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['1988','1989'])
    parser.add_option("-i","--initial_num_per_class", action="store", type="int", dest="initial_num_per_class", help="initial_num_per_class", default=5)
    parser.add_option("-g","--gridsearch_interval", action="store", type="int", dest="gridsearch_interval", help="perform gridsearch every N iterations", default=10)
    parser.add_option("-c","--num_candidate", action="store", type="int", dest="num_candidate", help="number of candidates selected from unlabelled pool each iteration", default=10)
    parser.add_option("-l","--loop", action="store", type="int", dest="loop", help="total number of iterations", default=50)
    parser.add_option("-s","--selection_strategy", action="store", type="string", dest="selection_strategy", help="selection strategy options are mostConfident and uncertainty", default = 'mostConfident')
    parser.add_option("-e","--encoding_method", action="store", type="string", dest="encoding_method", help="text representation technique options are roberta-base and PV-TD", default = 'roberta-base')


    (options, _) = parser.parse_args()


    random_states = [int(number) for number in options.random_seeds]

    text_rep = options.encoding_method
  

    dataset = options.dataset_name
    dir_neg = dataset + '_neg.csv'
    dir_pos = dataset + '_pos.csv'

    representations_neg = genfromtxt(f"./datasets/{text_rep}_data/{dir_neg}", delimiter=',')
    representations_pos = genfromtxt(f"./datasets/{text_rep}_data/{dir_pos}", delimiter=',')
    ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)
    labels = np.array([0]*len(representations_neg)+[1]*len(representations_pos))


    
    
    for seed in random_states:
        raw_results = []
        raw_metrics = []
        entropy = []
        permutations = []
        
        for loop in range(options.loop):
            print('processing ',loop,'th loops---------------')

            if loop==0:
                
                selection_method = options.selection_strategy
               
                
                args = easydict.EasyDict({
                        "n_seedsamples":options.initial_num_per_class,
                        "initial_random_seed":seed,
                        "initial": True,
                        "TEXT_emb": ulti_representations,
                        "LABEL_emb": labels,
                        "gridsearch":True,
                        "loop":loop,
                        "gridsearch_interval":options.gridsearch_interval

                })

                raw_predict,raw_metric,y_prob,y_pred,permutation,best_params_ = active_process_svm(args)



                permutation = sample_candidates(num_candidate=options.num_candidate,permutation=permutation,selection_method=selection_method,y_prob=y_prob,loop=loop)
                permutations.append(permutation)
                
                raw_results.append(raw_predict)
                raw_metrics.append(raw_metric)
                
                df = pd.DataFrame()
                df_p = pd.DataFrame()
                df_m = pd.DataFrame()

                save_dir = f"./outputs/{dataset}/"

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                
                df[seed] = [item for item in raw_results]
                df.to_csv(f"./outputs/{dataset}/raw_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
                df_p[seed] = [permutation for permutation in permutations]
                df_p.to_csv(f"./outputs/{dataset}/permutation_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
                df_m[seed] = [metric for metric in raw_metrics]
                df_m.to_csv(f"./outputs/{dataset}/metrics_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
        
            else:

                
                    
                  
                args = easydict.EasyDict({
                        "n_seedsamples":options.initial_num_per_class,
                        "initial_random_seed":seed,
                        "initial": False,
                        "TEXT_emb": ulti_representations,
                        "LABEL_emb": labels,
                        "gridsearch":True,
                        "loop":loop,
                        "gridsearch_interval":options.gridsearch_interval,
                        "permutation":permutation,
                        "best_params_":best_params_

                })

                raw_predict,raw_metric,y_prob,y_pred,permutation,best_params_ = active_process_svm(args)

                permutation = sample_candidates(num_candidate=options.num_candidate,permutation=permutation,selection_method=selection_method,y_prob=y_prob,loop=loop)
                permutations.append(permutation)
                
                raw_results.append(raw_predict)
                raw_metrics.append(raw_metric)
                
                df = pd.DataFrame()
                df_p = pd.DataFrame()
                df_m = pd.DataFrame()
                
                df[seed] = [item for item in raw_results]
                df.to_csv(f"./outputs/{dataset}/raw_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
                df_p[seed] = [permutation for permutation in permutations]
                df_p.to_csv(f"./outputs/{dataset}/permutation_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
                df_m[seed] = [metric for metric in raw_metrics]
                df_m.to_csv(f"./outputs/{dataset}/metrics_{text_rep}_{seed}_{selection_method}_result.csv",index=False)
                
                
                       

        print('finish_______________',seed)
               
                
                
            






