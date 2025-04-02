


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
import gzip
from collections import Counter
from sklearn.datasets import dump_svmlight_file
import time
from sklearn.model_selection import GridSearchCV
from src.model.traditionalhelpers import base_estimators, default_params
import numpy as np

from scipy.special import expit  # Sigmoid function

#from tqdm import tqdm
#from torch.optim import Adam
#from transformers import AutoModelForSequenceClassification
#from transformers import AutoTokenizer
#from src.model.slmdatahandle import prepare_training_datasets, prepare_inference_datasets, prep_data #

class TraditionalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_config, dataset):

        self.model_config = model_config
        #self.model_name = self.model_config.model_name
        self.model_name = self.model_config.model_tag
        self.dataset = dataset

        self.model = base_estimators[self.model_name]
        self.params = default_params[self.model_name].copy()

        #if self.model_name == 'svm' or self.model_name == 'lsvm':
        #    self.params['max_iter'] = args['max_iter']

        self.model.set_params(**self.params)

        # Por fold
        #self.micro_validation = None
        #self.macro_validation = None

        self.grid_time = 0
        self.train_time = 0
        self.test_time = 0

        #self.GridSearchCVvalues = self.args['GridSearchCVvalues']

        print(self.model)

    def fit(self, X, y=None):

        #Possibilitando executar o cv para datasets ext pequenos
        #counter = Counter(y)
        #mininum = min(counter, key=counter.get)
        #if counter[mininum] < 10 and self.dataset not in ['webkb', '20ng', 'acm', 'reut', 'reut90']:
        #    print(f"Adjusting CV value to {counter[mininum]}")
        #    self.args['cv'] = counter[mininum]
        #    print(self.args)

        #if self.args['cv'] > 1:
        if 'cv' in self.model_config.training_args:
            #t_init = time.time()
            if 'n_jobs' in self.model_config.training_args:
                n_jobs = self.model_config.training_args.n_jobs
            else:
                n_jobs = -1

            #tunning = default_tuning_params[self.args['name_class']]
            tunning = [dict(self.model_config.training_args.tuning_params)]

            t_init = time.time()

            gs = GridSearchCV(self.model, tunning,
                                n_jobs=n_jobs,
                                # refit=False,
                                #cv=self.args['cv'],
                                cv=self.model_config.training_args.cv,
                                verbose=1,
                                scoring='f1_micro')

            gs.fit(X, y)
            print(gs.best_score_, gs.best_params_)

            
            self.model.set_params(**gs.best_params_)
            #if self.GridSearchCVvalues:
            #    self.micro_validation = gs.cv_results_[
            #        'mean_test_f1_micro'][gs.best_index_]
            #    self.macro_validation = gs.cv_results_[
            #        'mean_test_f1_macro'][gs.best_index_]

            self.grid_time = time.time() - t_init

            # self.args['best_param_class'].append(gs.best_params_)

        print(self.model)
        self.model = clone(self.model)

        # fit and predict
        print('Fitting')
        t_init = time.time()
        self.model.fit(X, y)
        self._time_to_train = time.time() - t_init
        self._time_to_train += self.grid_time
        #if self.args['name_class'] != 'nc':
        #    # calibrator pro predict proba
        #    self.calibrator = CalibratedClassifierCV(self.model, cv='prefit')
        #    self.calibrator.fit(X, y)

        return self

    def predict(self, X, y=None):
        print('Predicting')
        t_init = time.time()
        self.y_pred = self.model.predict(X)
        self._time_to_predict = time.time() - t_init
        return self.y_pred

    def predict_proba(self, X, y=None):
        print('Predicting')
        t_init = time.time()
        
        if self.model_name == 'lsvm':
            y_margins = self.model.decision_function(X)
            proba = expit(y_margins)
            
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])

        else:
            proba = self.model.predict_proba(X)

        #assert np.all(self.model.predict(X) == np.argmax(proba, axis=-1))
        
        self._time_to_predict = time.time() - t_init
        return proba
        
        #if self.args['name_class'] == 'nc':
        #    return self.model.predict_proba(X)
        #return self.calibrator.predict_proba(X)

    #def save_proba(self, X, y, f, tipo):
    #    with gzip.open(self.args['finaloutput']+"proba_"+tipo+"_"+str(f)+".gz", 'w') as filout:
    #        dump_svmlight_file(X, y, filout, zero_based=False)
    #
    #def save_model(self, f):
    #    pickle.dump(self.model, open(
    #        self.args['finaloutput']+"model_"+str(f), 'wb'))
