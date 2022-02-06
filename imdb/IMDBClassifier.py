import os, re, wget, tarfile, time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import transformers

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

### Define DistilBert Model Definitions
model_checkpoint = 'distilbert-base-uncased'
transformer_model = transformers.AutoModel.from_pretrained(model_checkpoint)
transformer_tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)
transformer_config = transformers.AutoConfig.from_pretrained(model_checkpoint)  

class ImdbData():
    # download the original IMDB dataset in tgz compressed format
    # os.system('wget https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz')
    
    # Untar the tgz file - It will create an imdb directory with a subdirectory for train and test data.  The train and test directory will have pos and neg subdirectories.
    # The pos directory will contain all the positive reviews text and the neg directory will contain all the negative reviews text.  There are 25,000 text files each in the train
    # and test data.  Of the 25,000 text files, 12,500 are positive review text files and 12,500 are negative review text files.  It is a balanced data set.  
    def __init__(self) -> None:
        data_source_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.data_file_path='/opt/ml/input/aclImdb_v1.tar.gz'
        self.data_folder='/opt/ml/input/aclImdb'
        if not self.check_if_file_exists(self.data_file_path):
            print('Start of data download')
            wget.download(url=data_source_url, out='/opt/ml/input')
            print('Download complete')
        else:
            print('Data file already exists. Not downloading again!')

        if not self.check_if_dir_exists(self.data_folder):
            startTime = time.time()
            tar = tarfile.open(self.data_file_path)
            print('Extracting all the files now...')
            tar.extractall('/opt/ml/input')
            tar.close()
            print('Done!') 
            total_time=time.time()-startTime
            print('Time Taken for extracting all files : ',total_time/60,'minutes')
        else:
            print('Data folder exists. Won\'t copy again!')    

    def check_if_file_exists(self, file):
        '''
        Checks if 'file' exists
        '''
        try:
            tarfh = tarfile.open(file)
            return True
        except FileNotFoundError:
            #print('Please make sure file: ' + file + ' is present before continuing')
            return False

    def check_if_dir_exists(self, directory):
        '''
        Checks if 'directory' exists
        '''
        return(os.path.isdir(directory))

    """
    Load file into memory
    """
    def load_file(self, filename):
        """
        Open the file as read only
        """
        file = open(filename, 'r')
        """
        Read all text
        """
        text = file.read()
        """
        Close the file
        """
        file.close()
        return text.strip()    

    ## Remove the HTML tags like <br> in the text
    def preprocess_reviews(self, review):
        #REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        NO_SPACE = ""
        SPACE = " "
        #reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
        review = REPLACE_WITH_SPACE.sub(SPACE, review)
    
        return review
 
    def get_data(self):
        """
        Reading train test directory
        """
        reviews_dict={'train':[],'test':[]}
        for split in ['train','test']:
            directory = os.path.join(self.data_folder,split)
            for sentiment in ['pos', 'neg']:
                data_folder=os.path.join(directory, sentiment)
                print('Data Folder : ',data_folder)
                for root, dirs, files in os.walk(data_folder):
                    for fname in files:
                        if fname.endswith(".txt"):
                            file_name_with_full_path=os.path.join(root, fname)
                            review=self.load_file(file_name_with_full_path)
                            clean_review=self.preprocess_reviews(review)
                            if split == 'train':
                                reviews_dict['train'].append(clean_review)
                            else:
                                reviews_dict['test'].append(clean_review)
        return reviews_dict

## Define the IMDBDataset
# custom dataset uses Bert Tokenizer to create the Pytorch Dataset
class ImdbDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.notes = examples['reviews']
        self.targets = examples['target']
        self.tokenizer = tokenizer
        self.max_len = max_len
                
    def __len__(self):
        return (len(self.notes))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        note = str(self.notes[idx])
        target = self.targets[idx]
        
        encoding = self.tokenizer.encode_plus(
            note,
            add_special_tokens=True,
            max_length=self.max_len,
            ## Token Type IDs not really needed for this case
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
        )    
        return {
            'label': target,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids']
        }

## Define IMDB DataModule class
class ImdbDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 4, **kwargs):
        super().__init__()
          
        # Defining batch size of our data
        self.batch_size = batch_size
          
        # Defining num_workers
        self.num_workers = num_workers

        # Defining Tokenizers
        self.tokenizer = transformer_tokenizer

        self.imdbdata = ImdbData()
    
    def prepare_data(self):
        reviews_dict = self.imdbdata.get_data()
        ## Creating dataframes from the list data.  The reviews are arranged in the order that the first 12,500 belong to positive reviews and the rest 12,500 belong to negative reviews.
        df_train_reviews_clean = pd.DataFrame(reviews_dict['train'], columns =['reviews'])
        df_train_reviews_clean['target'] = np.where(df_train_reviews_clean.index<12500,1,0)
        df_test_reviews_clean = pd.DataFrame(reviews_dict['test'], columns =['reviews'])
        df_test_reviews_clean['target'] = np.where(df_test_reviews_clean.index<12500,1,0)

        # Shuffling the rows in both the train and test data.  This is very important before using the data for training.
        df_train_reviews_clean = df_train_reviews_clean.sample(frac=1).reset_index(drop=True)
        df_test_reviews_clean = df_test_reviews_clean.sample(frac=1).reset_index(drop=True)

        # breaking the train data into training and validation
        self.df_train, self.df_valid = train_test_split(df_train_reviews_clean, test_size=0.25, stratify=df_train_reviews_clean['target'])
        self.df_train.reset_index(drop=True, inplace=True)
        self.df_valid.reset_index(drop=True, inplace=True)
        self.df_test = df_test_reviews_clean
  
    def setup(self, stage=None):
        # Loading the dataset
        ### The Max token length of Input for a Bert Model is 512 tokens.  Setting it here just below the max.  The text will be truncated to 500 tokens
        self.train_dataset = ImdbDataset(self.df_train, tokenizer=transformer_tokenizer, max_len=500)
        self.val_dataset = ImdbDataset(self.df_valid, tokenizer=transformer_tokenizer, max_len=500)
        self.test_dataset = ImdbDataset(self.df_test, tokenizer=transformer_tokenizer, max_len=500)
  
    def custom_collate(self,features):
        features = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        return features    
  
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle = True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

### Define IMDB Moodel
class ImdbModel(torch.nn.Module):
    def __init__(self,
                 num_labels: int = 2,
                 **kwargs):
        super().__init__()
     
        self.num_labels = num_labels
        self.bert = transformer_model
        self.tokenizer = transformer_tokenizer
        
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(self.bert.config.seq_classif_dropout)

        # relu activation function
        self.relu =  torch.nn.ReLU()

        # freeze the layers of Bert for training if needed so that the embeddings of all layers of Bert are not changed
        # Tried this and got a much higher loss after 1 epoch compared to not freezing
        #for param in self.bert.parameters():
        #  param.requires_grad = False

    
    def forward(self, batch):
      
        outputs = self.bert(input_ids=batch['input_ids'], \
                         attention_mask=batch['attention_mask'])

        ## Output from last Hidden layer
        hidden_state = outputs[0]  # (bs, seq_len, dim)
        ## Output of CLS token - considered to represent the hidden state of entire sentence
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        ## Send the hidden state of CLS token thru Linear, Relu and dropout layers
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        return logits
    
    def get_outputs(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)
        return outputs

### IMDB Classifier Class
class ImdbClassifier(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.0001 * 8, **kwargs):
        super().__init__()
        self.save_hyperparameters('learning_rate','max_epochs')
        self.model = ImdbModel() 

    def training_step(self, batch, batch_nb):
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))
        
        # logs
        self.log_dict({'train_loss':loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        # fwd
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), batch['label'].cpu())
        val_acc = torch.tensor(val_acc)
        
        # logs
        self.log_dict({'val_loss':loss,'val_acc':val_acc}, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_nb):
        y_hat = self.model(batch)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.model.num_labels), batch['label'].view(-1))
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), batch['label'].cpu())
        
        # logs
        self.log_dict({'test_loss':loss,'test_acc':test_acc}, prog_bar=True)
  
        return loss
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self.model(batch)    

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.learning_rate, eps=1e-08)
        scheduler = {
        'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-5, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()), epochs=self.hparams.max_epochs),
        'interval': 'step'  # called after each training step
        } 
        
        return [optimizer], [scheduler]
       
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser])

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'train_val_data'), type=str)

        # training params (opt)
        parser.add_argument('--learning_rate', default=2e-5, type=float, help = "type (default: %(default)f)")
        return parser

