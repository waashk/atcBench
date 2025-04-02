import torch
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import copy
import numpy as np

from datasets import Dataset
#from src.main.python.model.llmConfig import tokenize_function

class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels=None, n_test=0):
		self.encodings = encodings
		self.labels = labels
		if self.labels:
			self.lenght = len(self.labels)
		else:
			self.lenght = n_test

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		if self.labels:
			item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return self.lenght

def duplicate_if_necessary(X, y):
	#q = []
	q1 = []

	y_set = list(sorted(list(set(y))))
	mydict = Counter(y)

	for i in mydict.keys():
		if mydict[i] == 1:
			q1.append(i)

	X_res = copy.copy(X)
	y_res = copy.copy(y)

	if len(q1)>0:
		print("duplicando")
		for q in q1:
			print("duplicando", q)
			#print(np.where(y_res == q))
			q2 = np.where(y_res == q)[0][0]
			#print(q2)
			y_res = np.append(y_res,y_res[q2])
			#print(len(y_res))
	
			#X_res.data = np.hstack((X_res.data,X_res[q2].data))
			#X_res.indices = np.hstack((X_res.indices,X_res[q2].indices))
			#X_res.indptr = np.hstack((X_res.indptr,(X_res[q2].indptr + X_res.nnz)[1:]))
			#X_res._shape = (X_res.shape[0]+1,X_res[q2].shape[1])
			X_res.append(X_res[q2])

	return X_res, y_res

def prep_data(X_train, y_train, X_val=None, y_val=None): 
	#if not os.path.exists(working_dir):
	#    os.makedirs(working_dir)
	X_train, y_train = duplicate_if_necessary(X_train, y_train)
	
	to_insert = []
	print(f"Max Class ={max(y_train)}")
	for i in range(max(y_train)):
		if i not in y_train:
			to_insert.append(i)

	for ti in to_insert:
			print(f"Adding empty document to class {ti}")
			X_train = X_train + [" "]*2
			y_train = np.append(y_train, [ti]*2)

	#print(min(y_train), max(y_train))
	#print(len(set(y_train)))
	#print(y_train[:8:-1])

	#X_train, y_train = create_if_necessary(X_train, y_train)

	if X_val:
		return X_train, y_train, X_val, y_val

	#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
	sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2018)
	for train_index, val_index in sss.split(X_train, y_train):
		continue

	X_train_new = [X_train[x] for x in train_index]
	y_train_new = [y_train[x] for x in train_index]
	X_val   = [X_train[x] for x in val_index]
	y_val   = [y_train[x] for x in val_index]

	#to_insert = []
	#for i in range(max(y_val)):
	#	if i not in y_val:
	#		print(f"Adding empty document to class {i}")
	#		to_insert.append(i)
	#print(min(y_val), max(y_val))

	return X_train_new, y_train_new, X_val, y_val


def prepare_training_datasets(X_train, y_train, X_val, y_val, tokenizer, max_len):

	X_train, y_train, X_val, y_val = prep_data(X_train, y_train, X_val, y_val)
 
	train_dataset = Dataset.from_dict({"text": X_train,
									       "label": y_train})
	train_dataset = train_dataset.map(tokenize_function, 
										batched=True, 
										fn_kwargs={"tokenizer":tokenizer, "max_length": max_len})
	
	val_dataset = Dataset.from_dict({"text": X_val,
										"label": y_val})
	val_dataset = val_dataset.map(tokenize_function, 
										batched=True, 
										fn_kwargs={"tokenizer":tokenizer, "max_length": max_len})
 
	return train_dataset, val_dataset


def prepare_inference_datasets(X_test, tokenizer, max_len):
	
    test_dataset = Dataset.from_dict({"text": X_test})
                                           #"label": y_test})
    test_dataset = test_dataset.map(tokenize_function, 
                                        batched=True, 
                                        fn_kwargs={"tokenizer":tokenizer, "max_length": max_len})
	
    return test_dataset


#preprocess_batch
def tokenize_function(dataset, tokenizer, max_length):
#def tokenize_function(dataset):
    """
    Tokenizes dataset 

    :param dataset: Dataset 
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        dataset["text"],
        max_length = max_length,
        truncation = True,
        padding = "max_length"
    )