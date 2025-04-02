
from sklearn.base import BaseEstimator, ClassifierMixin
from src.utils.misc import Documents, Classes
#from src.main.python.model.data import prep_data, prepare_training_datasets #CustomDataset
from src.model.llmhelpers import create_bnb_config, create_peft_config, compute_metrics
from src.model.llmdatahandle import prepare_training_datasets, prepare_inference_datasets

from transformers import TrainingArguments
from transformers import Trainer

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import numpy as np
import time

class LLMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_config, dataset):

        self.model_config = model_config
        self.dataset = dataset
        self.model_name = model_config['model_name'] #deepmethod
        self.model_tag = model_config['model_tag'] 
        #self.pretrained_models_path = pretrained_models_path
        self.max_len = model_config['training_args']['max_len']
        self.learning_rate = model_config['training_args']['lr']
        self.batch_size = model_config['training_args']['batch_size']
        self.num_max_epochs = model_config['training_args']['n_epochs']
        
        if 'lora' in model_config:
            self.lora_alpha = model_config['lora']['alpha']
            self.lora_dropout = model_config['lora']['dropout']
            self.lora_r = model_config['lora']['r']

        self.logging_dir = f"logs/{self.model_tag}/{self.dataset}/"
        print(self.logging_dir)

    def set_model(self):

        return

    def load_model(self, model_name, num_training_labels):
        """
        Loads model and model tokenizer

        :param model_name: Hugging Face model name
        :param bnb_config: Bitsandbytes configuration
        """

        # Get number of GPU device and set maximum memory
        #n_gpus = torch.cuda.device_count()
        #print(n_gpus)
        #max_memory = f'{40960}MB'

        print("Setting bnb config...")
        # Load bitsandbytes configuration
        # Activate 4-bit precision base model loading
        peft_config = self.model_config['peft_config']
        bnb_config = create_bnb_config(load_in_4bit = peft_config['4bit']['load_in_4bit'], 
                                        bnb_4bit_use_double_quant = peft_config['4bit']['bnb_4bit_use_double_quant'], 
                                        bnb_4bit_quant_type = peft_config['4bit']['bnb_4bit_quant_type'] )
        #bnb_config = create_bnb_config()

        print("Loading model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_training_labels, 
            torch_dtype="auto", 
            device_map="auto", 
            quantization_config = bnb_config,
            #load_in_8bit=True
        )

        #@TODO Tem varias variacoes aqui
        # Load model tokenizer with the user authentication token
        #tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)


        #if model_name == "meta-llama/Llama-2-7b-hf" or  \
        #    model_name == "meta-llama/Meta-Llama-3-8B" or  \
        #    model_name == "meta-llama/Meta-Llama-3.1-8B" or \
        #    model_name == "mistralai/Mistral-7B-v0.3" or \
        #    model_name == "google/gemma-2-9b" or \
        #    model_name == "microsoft/phi-4"  or \
        #    model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":

        # Set padding token as EOS token
        tokenizer.pad_token = tokenizer.eos_token 
        model.config.pad_token_id = model.config.eos_token_id 

        print("Model loaded!")

        return model, tokenizer

    def fit(self, X_train: Documents, y_train: Classes, 
            X_val: Documents = None, y_val: Classes = None):

        self._time_to_train = time.time()

        # Load model from Hugging Face with model name
        self.model, self.tokenizer = self.load_model(self.model_name, 
                                                     num_training_labels=len(list(set(y_train))))

        print(self.model_name)

       
        #Load datasets and split them into train/validation
        train_dataset, val_dataset = prepare_training_datasets(X_train, 
                                                               y_train, 
                                                               X_val, 
                                                               y_val, 
                                                               self.tokenizer, 
                                                               self.max_len)

        training_args = TrainingArguments(output_dir=self.logging_dir,                    
                    per_device_train_batch_size=self.batch_size, 
                    per_device_eval_batch_size=self.batch_size,
                    num_train_epochs=self.num_max_epochs, 
                    #optim="adafactor",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    learning_rate = self.learning_rate)
        
        if 'lora' in self.model_config:
            print('Configuring perf....')
            # Create PEFT configuration for these modules and wrap the model to PEFT
            # Creates Parameter-Efficient Fine-Tuning configuration for the model
            peft_config = create_peft_config(r=self.lora_r, 
                                            lora_alpha=self.lora_alpha, 
                                            lora_dropout=self.lora_dropout, 
                                            bias=self.model_config['peft_config']['bias'],
                                            task_type=self.model_config['peft_config']['task_type'])

                                            #bias = "none", 
                                            #task_type= "SEQ_CLS")  #"CAUSAL_LM", 
                                            #,target_modules="all-linear"
            self.model.add_adapter(peft_config, adapter_name="adapter_1")

        print(self.model)
        
        self.trainer = Trainer( 
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics     
        )

        self.trainer.train()

        self._time_to_train = time.time() - self._time_to_train


    def predict(self, X_test):

        self._time_to_predict = time.time()

        #test_dataset = Dataset.from_dict({"text": X_test})
        #                                   #"label": y_test})
        #test_dataset = test_dataset.map(tokenize_function, 
        #                                  batched=True, 
        #                                  fn_kwargs={"tokenizer":self.tokenizer, "max_length": self.max_len})
        
        test_dataset = prepare_inference_datasets(X_test, self.tokenizer, self.max_len)

        predictions = self.trainer.predict(test_dataset)
        predictions = np.argmax(predictions.predictions, axis=-1)

        self._time_to_predict = time.time() - self._time_to_predict

        return predictions
    
    def predict_proba(self, X_test):

        self._time_to_predict = time.time()

        #test_dataset = Dataset.from_dict({"text": X_test})
        #                                   #"label": y_test})
        #test_dataset = test_dataset.map(tokenize_function, 
        #                                  batched=True, 
        #                                  fn_kwargs={"tokenizer":self.tokenizer, "max_length": self.max_len})
        
        test_dataset = prepare_inference_datasets(X_test, self.tokenizer, self.max_len)

        predictions = self.trainer.predict(test_dataset)
        predictions = predictions.predictions.tolist()
        #predictions = np.argmax(predictions.predictions, axis=-1)

        self._time_to_predict = time.time() - self._time_to_predict

        return predictions