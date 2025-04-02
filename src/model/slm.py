
from sklearn.base import BaseEstimator, ClassifierMixin
from src.utils.misc import Documents, Classes
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from src.model.slmdatahandle import prepare_training_datasets, prepare_inference_datasets, prep_data #

class SLMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_config, dataset):

        self.model_config = model_config
        self.dataset = dataset
        self.model_name = model_config['model_name'] #deepmethod
        self.model_tag = model_config['model_tag'] #
        #self.models_path = models_path
        self.max_len = model_config['training_args']['max_len']
        self.learning_rate = model_config['training_args']['lr'] #learning_rate
        self.batch_size = model_config['training_args']['batch_size'] #batch_num
        self.num_max_epochs = model_config['training_args']['num_max_epochs'] #max_iter
        self.max_patience = model_config['training_args']['patience'] #max_patience

        self.weight_decay_rate = model_config['training_args']['weight_decay_rate']
        self.min_val_epoch_impro_delta = model_config['training_args']['min_val_epoch_impro_delta']
        self.max_grad_norm = model_config['training_args']['max_grad_norm']

        #weight_decay_rate: float = 0.01, 
        #learning_rate: float = 5e-5, 
        #save_rep: bool = False,
        #min_val_epoch_impro_delta: float = 1e-4,

        self.full_finetuning = True
        self.epoch_id = 0

        self.logging_dir = f"logs/{self.model_tag}/{self.dataset}/"
        print(self.logging_dir)

    def load_model(self, model_name, num_training_labels):

        print("Loading model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_training_labels, 
            torch_dtype="auto", 
            device_map="auto", 
        )

        #@TODO Tem varias variacoes aqui
        # Load model tokenizer with the user authentication token
        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  do_lower_case=False,
                                                  max_length = self.max_len)

        # Set padding token as EOS token
        #@TODO
        if self.model_tag == 'roberta' or self.model_tag == 'bart':
            tokenizer.add_prefix_space = True

        print("Model loaded!")

        return model, tokenizer


    def fit(self, X_train: Documents, y_train: Classes, X_val: Documents = None, y_val: Classes = None):
        """Fine-tuning of the pre-trained XLNet model.

        Parameters
        ----------
        :param X: Documents
            Documents for training.
        :param y: Classes
            Classes for each document in training.
        """
        gpu_id = 0
        self.device = torch.device(f'cuda:{gpu_id}')

        X_train, y_train, X_val, y_val = prep_data(X_train, y_train, X_val, y_val)

        #@TODO remover, so test
        #X_train, y_train = X_train[:1000], y_train[:1000] 

        #self.num_classes = len(list(set(y)))
        self.num_classes = len(list(set(y_train)))
        print(self.num_classes)

        #self.set_model() 
        self._time_to_train = time.time()

        # Load model from Hugging Face with model name
        self.model, self.tokenizer = self.load_model(self.model_name, 
                                                     num_training_labels=len(list(set(y_train))))

        print(self.model_name)

        # Send pre-trained model to GPU
        self.model.to(self.device)

        self.training_loss = []
        self.validation_loss = []
        patience = 0
        best_loss = None
        
        data_loader_train, data_loader_val = prepare_training_datasets(X_train, y_train, 
                                                                       X_val, y_val, 
                                                                       self.tokenizer, self.max_len, 
                                                                       self.batch_size)

        # Training the model
        optimizer = self._set_optimizer()

        #for epoch_id in range(self.epochs):
        #for epoch_id in range(10):
        #while True:
        while self.epoch_id < self.num_max_epochs:
            # Info
            self.epoch_id += 1

            self.model.train()
            #self.model.train(False)

            #logging.info(f'epoch: {self.epoch_id}')
            print(f'epoch: {self.epoch_id}')

            tr_loss = 0
            nb_tr_steps = 0
            #for step, batch in tqdm(enumerate(data_loader_train), desc="Train"):
            #for step, batch in tqdm(enumerate(data_loader_train)):
            for batch in tqdm(data_loader_train, desc="Train"):
                # Add batch to GPU
                #batch = tuple(t.to(self.device) for t in batch)
                #b_input_ids, b_input_mask, b_segs, b_labels = batch

                # Forward pass
                '''
                all labels=labels
                'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
                'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
                'transfoxl': input_ids=input_ids
                '''
                batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
                outputs = self.model(**batch)
                loss, logits = outputs[:2]

                # Backward pass
                loss.backward()

                # Track train loss
                tr_loss += loss.item()
                nb_tr_steps += 1

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=self.max_grad_norm)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

            # Print train loss per epoch
            self.training_loss.append(tr_loss / nb_tr_steps)
            #logging.info(f'train loss: {tr_loss / nb_tr_steps:.4E}')
            print(f'train loss: {tr_loss / nb_tr_steps:.4E}')

            #Calcular loss na validacao aqui
            self.model.eval()

            vl_loss = 0
            nb_vl_steps = 0
            #for step, batch in tqdm(enumerate(data_loader_val), desc = "Val"):
            for batch in tqdm(data_loader_val, desc = "Val"):
                # Add batch to GPU
                #batch = tuple(t.to(self.device) for t in batch)
                #b_input_ids, b_input_mask, b_segs, b_labels = batch

                with torch.no_grad():
                    # Forward pass
                    '''
                    all labels=labels
                    'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
                    'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
                    'transfoxl': input_ids=input_ids
                    '''
                    batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
                    outputs = self.model(**batch)
                    loss, logits = outputs[:2]

                vl_loss += loss.item()
                nb_vl_steps += 1

            dev_loss = vl_loss / nb_vl_steps
            
            if best_loss is None or \
                dev_loss + self.min_val_epoch_impro_delta < best_loss:
                
                best_loss = dev_loss
                print('val best loss updated: {:.4f}'.format(best_loss))
                #removed for big datasets: now patience dont need to be consective
                #patience = 0
            else:
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                
                new_lr = optimizer.param_groups[0]['lr']/2
                #optimizer.set_learning_rate(new_lr)
                for g in optimizer.param_groups:
                    g['lr'] = new_lr
                                
                print('patience #{}: reducing the lr to {}'.format(patience, new_lr))
                if patience == self.max_patience:
                    break
                patience+=1

            # Print train loss per epoch
            #logging.info(f'val loss: {dev_loss:.4E}')
            print(f'val loss: {dev_loss:.4E}')
            self.validation_loss.append(dev_loss)

        #if self.save_model: 
        #    self.model.save_pretrained(f'{self.out_dir}')

        self._time_to_train = time.time() - self._time_to_train

        return self


    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)

    def predict_proba(self, X: Documents):
        """Class probability prediction for new documents (using the fitted model).

        Parameters
        ----------
        :param X: Documents
            Documents for prediction.

        Returns
        ----------
        :return y_pred: numpy.ndarray
            Class probability predictions for each document in X.
        """
        # Generating test DataLoader
        #data_loader = self._generate_data_loader(X=X, partition='test') #training=False)
        #test_encodings = self.tokenizer(X, truncation=True, padding='max_length', max_length=self.max_len)
        #test_dataset = CustomDataset(test_encodings, n_test = len(X))
        #sampler_test = SequentialSampler(test_dataset)
        #data_loader_test = DataLoader(test_dataset, sampler = sampler_test, batch_size=self.batch_num, drop_last=False)
        self._time_to_predict = time.time()

        data_loader_test = prepare_inference_datasets(X, self.tokenizer, self.max_len, self.batch_size)

        # Evalue loop
        self.model.eval()

        #y_pred = []
        torch_logits = []
        #for step, batch in tqdm(enumerate(data_loader), desc="Test"):
        for batch in tqdm(data_loader_test, desc="Test"):
            #batch = tuple(t.to(self.device) for t in batch)
            #b_input_ids, b_input_mask, b_segs = batch

            with torch.no_grad():
                '''
                'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
                'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
                'transfoxl': input_ids=input_ids
                '''
                batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs[0]


            # Predictions
            logits = logits.detach().cpu().numpy()
            #y_pred.append(logits)
            torch_logits.append(logits)

        #y_pred = np.concatenate(y_pred)
        torch_logits = np.concatenate(torch_logits)
        #print(torch_logits[:10])

        proba = self.softmax(torch_logits)
        #print(proba[:10])

        self._time_to_predict = time.time() - self._time_to_predict

        #return y_pred
        return proba
    
    #@TODO REFAZER
    def representation(self, X: Documents):
        """Prediction for new documents (using the fitted model).

        Parameters
        ----------
        :param X: Documents
            Documents for prediction.

        Returns
        ----------
        :return y_pred: numpy.ndarray
            Predictions for each document in X.
        """
        # Generating DataLoader
        #data_loader = self._generate_data_loader(X=X, partition="test") #training=False)

        #test_encodings = self.tokenizer(
        #    X, truncation=True, padding='max_length', max_length=self.max_len)
        #test_dataset = CustomDataset(test_encodings, n_test=len(X))
        #sampler_test = SequentialSampler(test_dataset)
        #data_loader_test = DataLoader(
        #    test_dataset, sampler=sampler_test, batch_size=self.batch_num, drop_last=False)
        ##data_loader_test = DataLoader(
        ##	test_dataset, shuffle=False, batch_size=self.batch_num, drop_last=False)

        data_loader_test = prepare_inference_datasets(X, self.tokenizer, self.max_len, self.batch_size)

        # Evalue loop
        self.model.eval()

        rep_list = []
        #for step, batch in tqdm(enumerate(data_loader), desc="Test"):
        for batch in tqdm(data_loader_test, desc="Representation"):
            #batch = tuple(t.to(self.device) for t in batch)
            #b_input_ids, b_input_mask, b_segs = batch

            with torch.no_grad():
                '''
                'xlnet':input_ids=b_input_ids, token_type_ids=b_segs, input_mask=b_input_mask
                'roberta' or 'gpt2': input_ids=input_ids, attention_mask=attention_mask
                'transfoxl': input_ids=input_ids
                '''
                batch = {k: v.type(torch.long).to(self.device) for k, v in batch.items()}
                if self.model_tag == 'bert':
                    #outputs = self.model(**batch)
                    #print(outputs)
                    outputs = self.model.bert(**batch)['pooler_output']
                if self.model_tag == 'roberta':
                    outputs = self.model.roberta(**batch)['last_hidden_state']
                if self.model_tag == 'bart':
                    outputs = self.model(**batch)
                    #print(len(outputs))
                    outputs = outputs.encoder_last_hidden_state
                

            outputs = outputs.cpu().detach().numpy().tolist()
            #doc = {'id' : ids_train[index_doc], 'bert' : outputs, 'label' : label.numpy().tolist()[0]}
            
            for out in outputs:
                
                if self.model_tag == 'roberta' or self.model_tag == 'bart':
                    out = np.mean(out, 0).tolist()

                rep_list.append(out)

        return rep_list

    def score(self, X, y, sample_weight=None):
        pass

    def _set_optimizer(self) -> Adam:
        """Setting the optimizer for the fit method.

        Returns
        ----------
        :return optimizer: Adam
            The Adam optimizer from torch.optim.
        """
        if self.full_finetuning:
            # Fine tune all layer parameters of the pre-trained model
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.weight_decay_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            # Only fine tune classifier parameters
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer]}
            ]

        optimizer = Adam(optimizer_grouped_parameters, lr=self.learning_rate)

        return optimizer