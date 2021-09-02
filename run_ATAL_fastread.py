
# coding: utf-8

# In[1]:


import numpy as np 
np.random.seed(2019)

from numpy import genfromtxt
import glob
import random as r
r.seed(2019)

import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)


from sklearn.model_selection import train_test_split
from util import d, here

import pandas as pd
from argparse import ArgumentParser

import random, tqdm, sys, math, gzip
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit,StratifiedKFold

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import fine_tuned
import inference
import time

import shutil


import easydict
from argparse import ArgumentParser
from optparse import OptionParser
from sklearn import metrics


def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
## show size of variables
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

## random over sampling instances belong to minority class in case of imbalanced dataset
def resample(X,Y,random_state=2019):
    X_reshape = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    rus = RandomOverSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_reshape, Y)
    X_back = np.reshape(X_res, (X_res.shape[0],X.shape[1],X.shape[2]))
    
    return X_back, y_res

## seed the initial data
def initial_seed_dataset(n_initial, Y,random_state):
    
    np.random.seed(random_state)
    
    df = pd.DataFrame()
    df['label'] = Y

    Samplesize = n_initial  #number of samples that you want       
    initial_samples = df.groupby('label', as_index=False).apply(lambda array: array.loc[np.random.choice(array.index, Samplesize, False),:])

    permutation = [index[1] for index in initial_samples.index.tolist()]
    
    print ('initial random chosen samples', permutation)
    
    return permutation

## compute the entropy
def compute_entropy(output):
    entropy = -(output[:,0]*np.log2(output[:,0]) + output[:,1]*np.log2(output[:,1]))
    return entropy


## sample the candidates for labelling
def sample_candidates(selection_method,permutation,y_prob,num_candidate,y_pred,dist):
    
    print('*'*20,'selection method ',selection_method,'*'*20)
    
        
        
    if selection_method == 'uncertainty':
       
        candidate_index = []
        
        for index in dist.argsort():
            if len(candidate_index) == num_candidate:
                break
            if index not in permutation:
                candidate_index.append(index)

        permutation = permutation+candidate_index
        
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
        
    print('*'*20,'num of training set ',len(permutation),'*'*20)   
        
    return permutation

## normalization instance
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



    
## read data from directory
def read_data(path):

    corpora = []
    for filename in os.listdir(path):

        df_temp = pd.read_csv(path+filename)

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len

## construct training text dataset for fine tuning language model, save in directory data/
def construct_train_set(dataset,permutation):
    print('constructing new text training set.....')
    corpora, class_one_len, class_two_len = read_data('./corpus_data/'+dataset+'/')

    texts = np.array(corpora[0]+corpora[1])
    labels = np.array([0]*class_one_len+[1]*class_two_len)

#     X_train, X_test, y_train, y_test = train_test_split(texts[permutation], labels[permutation], test_size=0.0, random_state=2019)
    X_train, y_train = texts[permutation], labels[permutation]
    
    train_df = pd.DataFrame({'id':range(len(X_train)),'label':y_train,'alpha':['a']*len(X_train),'text':X_train})
#     dev_df = pd.DataFrame({'id':range(len(X_test)),'label':y_test,'alpha':['a']*len(X_test),'text':X_test})
    if not os.path.exists(f"ATAL_cache/data/"):
        os.mkdir(f"ATAL_cache/data/")

    train_df.to_csv(f"ATAL_cache/data/train.tsv", sep='\t', index=False, header=False)
#     dev_df.to_csv('./ATAL_results/data/dev.tsv', sep='\t', index=False, header=False)






# ## Part 3. Active Learning Process




def active_process(arg):
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
        Y_train_ori = arg.LABEL_emb[permutation]

        X_val = arg.TEXT_emb
        Y_val = arg.LABEL_emb
        bin_count_ori = np.bincount(Y_train_ori)
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
        au_permutation = arg.au_permutation
       
        
        X_train = arg.TEXT_emb[au_permutation]
        Y_train = arg.LABEL_emb[au_permutation]
        Y_train_ori = arg.LABEL_emb[permutation] 

        X_val = arg.TEXT_emb
        Y_val = arg.LABEL_emb
        
        bin_count_ori = np.bincount(Y_train_ori)
        unique_ori = np.unique(Y_train_ori)
        print (
        'training set size before agressive undersampling:',
        Y_train_ori.shape[0],
        'unique(labels):',
        unique_ori,
        'label counts:',
        bin_count_ori
        )
        
        bin_count = np.bincount(Y_train)
        unique = np.unique(Y_train)
        print (
        'training set size after agressive undersampling:',
        Y_train.shape[0],
        'unique(labels):',
        unique,
        'label counts:',
        bin_count
        )

        print('Number of training examples ', len(Y_train))


    
    
#     normalizer = Normalize()
#     X_train, X_val = normalizer.normalize(X_train, X_val) 
        
    
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
    y_eval_prob_pos = np.array(y_pred_prob)[:,1]
#     y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred = classifier.predict(X_val)
    y_true = Y_val
    
    numerators = classifier.decision_function(X_val)
    try:
        w_norm = np.linalg.norm(classifier.best_estimator_.coef_)
    except Exception:
        w_norm = np.linalg.norm(classifier.coef_)

    dist = abs(numerators) / w_norm
   

    y_true_remain = np.delete(np.array(y_true),permutation)
    y_pred_remain = np.delete(np.array(y_pred),permutation)
    y_eval_prob_pos_remain = np.delete(np.array(y_eval_prob_pos),permutation)
        
    print(classification_report(y_true_remain,y_pred_remain,labels=[0,1]))
   
    tn, fp, fn, tp = confusion_matrix(y_true_remain, y_pred_remain, labels=[0,1]).ravel()
    print('TP_H',bin_count[1],' TN_H',bin_count[0], ' TP_M',tp, ' TN_M',tn, ' FP_M', fp, ' FN_M',fn)
    acc = accuracy_score(y_true_remain, y_pred_remain)
    f_score = f1_score(y_true_remain,y_pred_remain,average='micro')
                
    fpr, tpr, thresholds = metrics.roc_curve(y_true_remain, y_eval_prob_pos_remain, pos_label=1)
    auc = metrics.auc(fpr, tpr)
     
    raw_result = {'TP_H':bin_count_ori[1],'TN_H':bin_count_ori[0], 'TP_M':tp, 'TN_M':tn, 'FP_M': fp, 'FN_M':fn}
    
    raw_metrics = {'ACC':acc,'micro_f_score': f_score, 'AUC':auc,'coverage':float(raw_result['TP_H']/(raw_result['TP_H']+raw_result['TP_M']+raw_result['FN_M']))}
    
    return raw_result, y_pred_prob, y_pred,permutation, best_params_,dist,raw_metrics


# In[ ]:



# Run the main function
# selection_method = 'certainty'
selection_method = 'uncertainty'
# selection_method = 'mostConfident'


if __name__ == "__main__":


    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -l loop -g gridsearch_interval -m max_len -c num_candidate -i initial_num_per_class')

    
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by token-level Roberta", default = 'animal_by_product')
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['1988','1989'])
    parser.add_option("-i","--initial_num_per_class", action="store", type="int", dest="initial_num_per_class", help="initial_num_per_class", default=5)
    parser.add_option("-g","--gridsearch_interval", action="store", type="int", dest="gridsearch_interval", help="perform gridsearch every N iterations", default=10)
    parser.add_option("-c","--num_candidate", action="store", type="int", dest="num_candidate", help="number of candidates selected from unlabelled pool each iteration", default=10)
    parser.add_option("-l","--loop", action="store", type="int", dest="loop", help="total number of iterations", default=50)
    parser.add_option("-a","--learning_rate", action="store", type="float", dest="learning_rate", help="learning rate for fine tuning", default=5e-5)

    
    (options, _) = parser.parse_args()


    text_rep = 'roberta-base'

    
    df_permutation = pd.DataFrame()
    
    ## using different random seeds to seed the initial dataset
    random_states = [int(number) for number in options.random_seeds]
    dataset = options.dataset_name
    

    for seed in random_states:
       
        raw_results = []
        raw_metrics = []
        entropy = []
        permutations = []
        
        for loop in range(options.loop):
            print('processing ',loop,'th loops---------------')

            ## first 20 loops are identical to normal active learning
            if loop==0:
                selection_method = 'uncertainty'
                
                dir_neg = dataset + '_neg.csv'
                dir_pos = dataset + '_pos.csv'
                representations_neg = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_neg, delimiter=',')
                representations_pos = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_pos, delimiter=',')
                ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)
                labels = np.array([0]*len(representations_neg)+[1]*len(representations_pos))
       
                parser = ArgumentParser()
                args = parser.parse_known_args()[0]
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

                raw_result,y_pred_prob,y_pred,permutation,best_params_,dist,raw_metric = active_process(args)


                permutation = sample_candidates(num_candidate=10,permutation=permutation,selection_method=selection_method,y_prob=y_pred_prob,y_pred=y_pred,dist=dist)
                permutations.append(permutation)

                    
                raw_results.append(raw_result)
                raw_metrics.append(raw_metric)

                df = pd.DataFrame()
                df_p = pd.DataFrame()
                df_m = pd.DataFrame()
                
                df[seed] = [item for item in raw_results]
                df.to_csv(f"./outputs/{dataset}/raw_atal_{seed}_fastread_result.csv",index=False)
                df_p[seed] = [permutation for permutation in permutations]
                df_p.to_csv(f"./outputs/{dataset}/permutation_atal_{seed}_fastread_result.csv",index=False)
                df_m[seed] = [metric for metric in raw_metrics]
                df_m.to_csv(f"./outputs/{dataset}/metrics_atal_{seed}_fastread_result.csv",index=False)

            else:
                
                neg_permutation = [index for index in permutation if labels[index]==0]
                pos_permutation = [index for index in permutation if labels[index]==1]
                num_pos_train = np.sum(labels[permutation])
                num_neg_train = len(permutation)-num_pos_train
                
                
                if num_pos_train>30:

                    selection_method = 'mostConfident'
     

                    if num_neg_train - num_pos_train>0:
                        print('-'*20,'start aggressive undersampling','-'*20)

                        au_neg = list(np.argsort(y_pred_prob[neg_permutation][:,0])[-num_pos_train:])
                        au_permutation = pos_permutation+au_neg

#                         print(au_permutation)


                    else:
                        au_permutation = permutation
                
                    ## start fine tuning
                    if loop==5:

                        ##empty previous model

                        files = glob.glob('./ATAL_cache/outputs/*')
                        for f in files:
                            shutil.rmtree(f)

                        construct_train_set(dataset,permutation)
                        ## fine tuning
                        checkpoint, accs = fine_tuned.fine_tuned(loop,dataset,'roberta-base',int(len(permutation)/4),options.learning_rate)

                        ## re-infer embeddings
                        inference.inference(dataset,checkpoint,loop)


                        ## save tuned embeddings to local
                        dir_neg = dataset + '_tuned_neg_%s.csv'%(loop)
                        dir_pos = dataset + '_tuned_pos_%s.csv'%(loop)
                        representations_neg = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_neg, delimiter=',')
                        representations_pos = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_pos, delimiter=',')

                        ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)

                        for key in accs.keys():
                            if key!= checkpoint:
                                shutil.rmtree(key)

                    #                     files = glob.glob('./ATAL_results/outputs/*')
                    #                     for f in files:
                    #                         if f!= checkpoint:
                    #                             shutil.rmtree(f)

                    elif (loop-5)%20==0 and loop<options.loop:
                        print('start fine tuning at loop ----------- ',loop,'----------')
                        construct_train_set(dataset,permutation)
                        checkpoint, accs = fine_tuned.fine_tuned(loop,dataset,checkpoint,int(len(permutation)/4),options.learning_rate)
                        inference.inference(dataset,checkpoint,loop)

                        dir_neg = dataset + '_tuned_neg_%s.csv'%(loop)
                        dir_pos = dataset + '_tuned_pos_%s.csv'%(loop)
                        representations_neg = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_neg, delimiter=',')
                        representations_pos = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_pos, delimiter=',')

                        ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)

                        for key in accs.keys():
                            if key!= checkpoint:
                                shutil.rmtree(key)

                    #                     files = glob.glob('./ATAL_results/outputs/*')
                    #                     for f in files:
                    #                         if f!= checkpoint:
                    #                             shutil.rmtree(f)

                    parser = ArgumentParser()
                    args = parser.parse_known_args()[0]
                    args = easydict.EasyDict({
                            "n_seedsamples":options.initial_num_per_class,
                            "initial_random_seed":seed,
                            "permutation": permutation,
                            "au_permutation":au_permutation,
                            "initial": False,
                            "TEXT_emb": ulti_representations,
                            "LABEL_emb": labels,
                            "best_params_":best_params_,
                            "loop":loop,
                            "gridsearch_interval":10
                    })

                    raw_result,y_pred_prob,y_pred,permutation,best_params_,dist,raw_metric = active_process(args)


                    permutation = sample_candidates(num_candidate=10,permutation=permutation,selection_method=selection_method,y_prob=y_pred_prob,y_pred=y_pred,dist=dist)
                    permutations.append(permutation)

                    
                    raw_results.append(raw_result)
                    raw_metrics.append(raw_metric)

                    df = pd.DataFrame()
                    df_p = pd.DataFrame()
                    df_m = pd.DataFrame()
                    
                    df[seed] = [item for item in raw_results]
                    df.to_csv(f"./outputs/{dataset}/raw_atal_{seed}_fastread_result.csv",index=False)
                    df_p[seed] = [permutation for permutation in permutations]
                    df_p.to_csv(f"./outputs/{dataset}/permutation_atal_{seed}_fastread_result.csv",index=False)
                    df_m[seed] = [metric for metric in raw_metrics]
                    df_m.to_csv(f"./outputs/{dataset}/metrics_atal_{seed}_fastread_result.csv",index=False)
                    
                else:
                
                    ## start fine tuning
                    if loop==5:

                        ##empty previous model

                        files = glob.glob('./ATAL_cache/outputs/*')
                        for f in files:
                            shutil.rmtree(f)

                        construct_train_set(dataset,permutation)
                        ## fine tuning
                        checkpoint, accs = fine_tuned.fine_tuned(loop,dataset,'roberta-base',int(len(permutation)/4),options.learning_rate)

                        ## re-infer embeddings
                        inference.inference(dataset,checkpoint,loop)


                        ## save tuned embeddings to local
                        dir_neg = dataset + '_tuned_neg_%s.csv'%(loop)
                        dir_pos = dataset + '_tuned_pos_%s.csv'%(loop)
                        representations_neg = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_neg, delimiter=',')
                        representations_pos = genfromtxt('./datasets/%s_data/'%(text_rep)+dir_pos, delimiter=',')

                        ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)

                        for key in accs.keys():
                            if key!= checkpoint:
                                shutil.rmtree(key)

                    #                     files = glob.glob('./ATAL_results/outputs/*')
                    #                     for f in files:
                    #                         if f!= checkpoint:
                    #                             shutil.rmtree(f)

                    elif (loop-5)%20==0 and loop<80:
                        print('start fine tuning at loop ----------- ',loop,'----------')
                        construct_train_set(dataset,permutation)
                        checkpoint, accs = fine_tuned.fine_tuned(loop,dataset,checkpoint,int(len(permutation)/4),1e-5)
                        inference.inference(dataset,checkpoint,loop)

                        dir_neg = dataset + '_tuned_neg_%s.csv'%(loop)
                        dir_pos = dataset + '_tuned_pos_%s.csv'%(loop)
                        representations_neg = genfromtxt('../datasets/%s_data/'%(text_rep)+dir_neg, delimiter=',')
                        representations_pos = genfromtxt('../datasets/%s_data/'%(text_rep)+dir_pos, delimiter=',')

                        ulti_representations = np.concatenate((representations_neg,representations_pos),axis=0)

                        for key in accs.keys():
                            if key!= checkpoint:
                                shutil.rmtree(key)

                    #                     files = glob.glob('./ATAL_results/outputs/*')
                    #                     for f in files:
                    #                         if f!= checkpoint:
                    #                             shutil.rmtree(f)
                    parser = ArgumentParser()
                    args = parser.parse_known_args()[0]
                    args = easydict.EasyDict({
                            "n_seedsamples":5,
                            "initial_random_seed":seed,
                            "permutation": permutation,
                            "au_permutation":permutation,
                            "initial": False,
                            "TEXT_emb": ulti_representations,
                            "LABEL_emb": labels,
                            "best_params_":best_params_,
                            "loop":loop,
                            "gridsearch_interval":10
                    })

                    raw_result,y_pred_prob,y_pred,permutation,best_params_,dist,raw_metric = active_process(args)


                    permutation = sample_candidates(num_candidate=10,permutation=permutation,selection_method=selection_method,y_prob=y_pred_prob,y_pred=y_pred,dist=dist)
                    permutations.append(permutation)

                    
                    raw_results.append(raw_result)
                    raw_metrics.append(raw_metric)

                    df = pd.DataFrame()
                    df_p = pd.DataFrame()
                    df_m = pd.DataFrame()
                    
                    df[seed] = [item for item in raw_results]
                    df.to_csv(f"./outputs/{dataset}/raw_atal_{seed}_fastread_result.csv",index=False)
                    df_p[seed] = [permutation for permutation in permutations]
                    df_p.to_csv(f"./outputs/{dataset}/permutation_atal_{seed}_fastread_result.csv",index=False)
                    df_m[seed] = [metric for metric in raw_metrics]
                    df_m.to_csv(f"./outputs/{dataset}/metrics_atal_{seed}_fastread_result.csv",index=False)              
              
        ## the tp_h, tn_h, tp_m, tn_m, fp_m, fn_m for each loop
        



        
        shutil.rmtree("./ATAL_cache/outputs/") 
        # times.append(time.time() - start_time)
        # print("--- %s seconds ---" % (time.time() - start_time))






