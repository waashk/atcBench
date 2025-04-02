
from sklearn import svm, ensemble
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid

base_estimators = {
    'svm': svm.SVC(),
    'lsvm': svm.LinearSVC(),
    'rf': ensemble.RandomForestClassifier(),
    'nb': MultinomialNB(),
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(),
    'nc': NearestCentroid(),
}


default_params = {
    'svm': 	{'kernel': 'linear', 'C': 1, 'verbose': False, 'probability': False,
             'degree': 3, 'shrinking': True,
             'decision_function_shape': None, 
             'tol': 0.001, 'cache_size': 25000, 'coef0': 0.0, 'gamma': 'auto',
             'class_weight': None, 'random_state': 1608637542, 	 'max_iter':1000000},
    'lsvm': {'loss': 'squared_hinge', 'C': 1, 'verbose': 0, 'intercept_scaling': 1,
             'fit_intercept': True, 'max_iter': 1000, 'penalty': 'l2',
             'multi_class': 'ovr', 'random_state': None, 'dual': False,
             'tol': 0.001, 'class_weight': None},
    'rf':  	{'warm_start': False, 'oob_score': False, 'n_jobs': 1, 'verbose': 0,
             'max_leaf_nodes': None, 'bootstrap': True, 'min_samples_leaf': 1,
             'n_estimators': 200, 'min_samples_split': 2,
             'min_weight_fraction_leaf': 0.0, 'criterion': 'gini',
             'random_state': None, 'max_features': 'auto', 'max_depth': None,
             'class_weight': None},
    'nb': 	{'alpha': 1.0, 'fit_prior': True, 'class_prior': None},
    'knn': 	{'n_neighbors': 30, 'weights': 'uniform', 'algorithm': 'auto', 'leaf_size': 30,  
             'p': 2, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': -1},
    'lr': {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True,
           'intercept_scaling': 1, 'class_weight': None, 'random_state': None, 
           #'solver': 'warn',
           'max_iter': 100, 
           #'multi_class': 'auto', 
           'verbose': 0, 'warm_start': False, 'n_jobs': -1, 'l1_ratio': None},
    'nc': {'metric': 'euclidean','shrink_threshold': None}, 
}


#default_tuning_params = {
#    'svm': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
#    'svmrbf': 	[{'C': 2.0 ** np.arange(-5, 15, 2)}],
#    'lsvm': [{'C': 2.0 ** np.arange(-5, 9, 2)}],
#    'rf': [{'criterion': ['entropy', 'gini'], 'n_estimators': [100, 200], 'max_features': ['sqrt', 'log2', 0.08, 0.15, 0.30]}],
#    'nb': [{'alpha': [1]}],
#    'nc': [{'metric': ['euclidean']}],
#    'knn': [{'n_neighbors': [10, 40, 70, 100]}],
#    'lr': [{'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 12), 'solver': ['liblinear']}]
#}