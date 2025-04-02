import os
import random

import numpy as np
#import torch
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import torch
import gc
import time

import pickle  
from sklearn.metrics import confusion_matrix, f1_score
import json
import socket

import gzip
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

def get_initialize_config(exp_dict: DictConfig):
    # delay import to reduce dependencies
    from ..utils.hydra import hydra_chdir
    hydra_chdir(exp_dict)

    #output_dir = exp_dict['config']['output_dir']
    #model = exp_dict['model']['model_tag']
    #dataset = exp_dict['data']['dataset']

    #exit()
    #print(type(exp_dict))
    print(exp_dict.paths.output_dir_full_path)
    makedir(exp_dict.paths.output_dir_full_path)

    #output_dir_full_path = f"{output_dir}/{model}/{dataset}/"
    #makedir(output_dir_full_path)

    return exp_dict
    
    #print(exp_dict.keys())
    #config = exp_dict["config"]
    #model_training_config = exp_dict["model"]
    
    exit()

    logging_dir=f"logs/{args.model}/{args.dataset}/",
    deepmethod = args.deepmethod,
    max_len = args.max_len,
    learning_rate = args.lr, 
    batch_size = args.batch_size,
    num_max_epochs = args.num_max_epochs
    
    exp_dict["init_dict"]["fp16"] = exp_dict["config"].get("fp16", False)
    config = exp_dict["config"]
    init_dict = exp_dict["init_dict"]
    if train:
        os.makedirs(exp_dict.config.checkpoint_dir, exist_ok=True)
        OmegaConf.save(config=exp_dict, f=os.path.join(exp_dict.config.checkpoint_dir, "config.yaml"))
        model_training_config = None
    else:
        if config.pretrained_no_yamlconfig:
            model_training_config = config
        else:
            model_training_config = OmegaConf.load(os.path.join(config["checkpoint_dir"], "config.yaml"))["config"]

        #if HF: need to update config (except for adapters...).
        #if not "adapter_name" in config and "hf_training" in config:
        if  "hf_training" in config:
            init_dict.model_type_or_dir=os.path.join(config.checkpoint_dir,"model")
            init_dict.model_type_or_dir_q=os.path.join(config.checkpoint_dir,"model/query") if init_dict.model_type_or_dir_q else None
                   
    return exp_dict, config, init_dict, model_training_config


def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def set_seed(seed):
    """see: https://twitter.com/chaitjo/status/1394936019506532353/photo/1
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seed setted.")

def set_seed_from_config(config):
    print(config)
    if "random_seed" in config:
        random_seed = config["random_seed"]
    else:
        random_seed = 123
    set_seed(random_seed)
    return random_seed

def get_raw_data_already_splitted(datadir, fold):

    #rootpath = config_data['DATA_DIR']
    #rootpath = config_data['DATA_DIR']

    df_train = pd.read_parquet(f"{datadir}/train_fold_{fold}.parquet")
    df_train['text'] = df_train['text'].str.replace(r'[\n\t\r\\]', ' ',regex=True)
    #doc_train_indexes = df_train.index.values
    X_train = list(df_train.text.values)
    #X_train = list(map(str.rstrip, X_train))
    y_train = list(df_train.label.values)
    
    df_test = pd.read_parquet(f"{datadir}/test_fold_{fold}.parquet")
    df_test['text'] = df_test['text'].str.replace(r'[\n\t\r\\]', ' ',regex=True)
    #doc_test_indexes = df_test.index.values
    X_test = list(df_test.text.values)
    #X_test = list(map(str.rstrip, X_test))
    y_test = list(df_test.label.values)

    y_encoder = LabelEncoder().fit(y_train)
    y_train = y_encoder.transform(y_train)
    #@TODO Voltar
    y_test = y_encoder.transform(y_test)
    #y_test = [int(_) for _ in y_test]

    return X_train, y_train, X_test, y_test

def get_data_parquet_file(filepath):

    df_data = pd.read_parquet(filepath)
    df_data['text'] = df_data['text'].str.replace(r'[\n\t\r\\]', ' ',regex=True)

    y_train = list(df_data.label.values)
    y_encoder = LabelEncoder().fit(y_train)
    y_train = y_encoder.transform(y_train)

    df_data.label = y_train

    return df_data


def get_encoded_representation(datadir, text_representation, fold):

    X_train, y_train = load_svmlight_file(
        f"{datadir}/{text_representation}/train{fold}.gz", dtype=np.float64)
    X_test, y_test = load_svmlight_file(
        f"{datadir}/{text_representation}/test{fold}.gz", dtype=np.float64)

    # Same vector size
    if (X_train.shape[1] > X_test.shape[1]):
        X_test, y_test = load_svmlight_file(
            f"{datadir}/{text_representation}/test{fold}.gz", dtype=np.float64, n_features=X_train.shape[1])
    elif (X_train.shape[1] < X_test.shape[1]):
        X_train, y_train = load_svmlight_file(
            f"{datadir}/{text_representation}/train{fold}.gz", dtype=np.float64, n_features=X_test.shape[1])

    y_encoder = LabelEncoder().fit(y_train)
    y_train = y_encoder.transform(y_train)
    #@TODO Voltar
    y_test = y_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

from sklearn.model_selection import StratifiedKFold

def generate_splits(df_data, n_folds):
    X = list(df_data.text.values)
    y = list(df_data.label.values)

    skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=42)

    splits_to_save = {c: [] for c in ["fold_id", "train_idxs", "test_idxs"]}
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        splits_to_save["fold_id"].append(i)
        splits_to_save["train_idxs"].append(train_index.tolist())
        splits_to_save["test_idxs"].append(test_index.tolist())

    df_splits = pd.DataFrame(data={
        "fold_id": splits_to_save["fold_id"],
        "train_idxs": splits_to_save["train_idxs"],  
        "test_idxs": splits_to_save["test_idxs"]
    })

    return df_splits




def load_splits(splits_filename):
    with open(splits_filename, "rb") as splits_file:
        #return pickle.load(splits_file)
        return pd.read_pickle(splits_file)
    
def save_splits(splits_df, filename):
    with open(filename, "wb") as split_file:
        pickle.dump(splits_df, split_file)

def split_raw_data(df_data, df_splits, fold):

    train_idxs = df_splits.loc[fold].train_idxs
    test_idxs = df_splits.loc[fold].train_idxs

    df_train = df_data.loc[train_idxs]
    df_test = df_data.loc[test_idxs]

    X_train = list(df_train.text.values)
    y_train = list(df_train.label.values)

    X_test = list(df_test.text.values)
    y_test = list(df_test.label.values)

    y_encoder = LabelEncoder().fit(y_train)
    y_train = y_encoder.transform(y_train)
    y_test = y_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test

def get_raw_data(config, fold):
    datadir = config.paths.DATA_DIR
    splits_dir = config.paths.SPLITS_DIR
    n_folds = config.data.nfolds

    #Open data.parquet
    df_data = get_data_parquet_file(f"{datadir}/data.parquet")
    
    if not os.path.exists(f"{splits_dir}/split_{n_folds}.pkl"):
        print("Generating splits...")
        df_splits = generate_splits(df_data, n_folds)
        print("Saving splits...")
        makedir(splits_dir)
        save_splits(df_splits, f"{splits_dir}/split_{n_folds}.pkl")
    else: 
        print("Loading splits...")
        df_splits = load_splits(f"{splits_dir}/split_{n_folds}.pkl")
        #print(df_splits)

    X_train, y_train, X_test, y_test = split_raw_data(df_data, df_splits, fold)

    return X_train, y_train, X_test, y_test




def getdata(config, fold):

    datadir = config.paths.DATA_DIR
    text_representation = config.model.text_representation


    if 'load_representation' in config.model and config.model.load_representation:
        
        if not (os.path.exists(f"{datadir}/{text_representation}/train{fold}.gz") and \
                os.path.exists(f"{datadir}/{text_representation}/test{fold}.gz") ) :
            
            raise Exception(f"Please, ensure you have train{fold}.gz and test{fold}.gz into the {datadir}/{text_representation}/ directory. Or change your config.model.text_representation to False.")
        
        X_train, y_train, X_test, y_test = get_encoded_representation(datadir, text_representation, fold) 
        return X_train, y_train, X_test, y_test
   
    if os.path.exists(f"{datadir}/train_fold_{fold}.parquet") and \
        os.path.exists(f"{datadir}/test_fold_{fold}.parquet") :
            X_train, y_train, X_test, y_test = get_raw_data_already_splitted(datadir, fold)
            return X_train, y_train, X_test, y_test

    else:
        
        if not os.path.exists(f"{datadir}/data.parquet"):
            raise Exception("Please, ensure you have the file data.parquet correcly specified.")

        X_train, y_train, X_test, y_test = get_raw_data(config, fold)
        
    if text_representation == 'tfidf':
        X_train, X_test = generate_tfidf(config, X_train, y_train, X_test, y_test)
    else:
        raise Exception(f"Only tfidf can be generated by this script. Please, ensure you have the {datadir}/{text_representation}/ directory correcly specified.")

    return X_train, y_train, X_test, y_test




def save_proba(file, X, y):
    with gzip.open(file, 'w') as filout:
        dump_svmlight_file(X, y, filout, zero_based=False)
          
def save_results(config, y_test, y_pred_proba, time_train, time_predict, fold):
    #args, clf, y_test, y_pred, y_pred_proba):

    outdir = config.paths.output_dir_full_path
    
    y_pred = np.argmax(y_pred_proba, axis=-1)
    pd.DataFrame(data={
         "y_test": y_test.tolist(),
         "y_pred": y_pred.tolist(),
    }).to_parquet(f"{outdir}/pred.fold_{fold}.parquet")

    #Measures
    micro = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
    macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    print(micro, macro)
    cm = confusion_matrix(y_test, y_pred).tolist()

    data = {
        "micro": micro,
        "macro": macro,
        "confusion_matrix": cm,
        "time_train": time_train,
        "time_predict": time_predict,
        "machine": socket.gethostname(),
    }

    with open(f"{outdir}/measures.fold_{fold}.json", 'w') as outfile:
        json.dump(data, outfile, indent=4)

    #if args.save_proba:
    print("Saving proba...")
    save_proba(f"{outdir}/proba.fold_{fold}.gz", y_pred_proba, y_test)
    print("Proba Saved")


def cleanMemory(clf):
    del clf.model
    if hasattr(clf, 'trainer'):
        del clf.trainer
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
