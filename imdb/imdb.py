import argparse
import os, pickle

# default pytorch import
import torch

# import lightning library
import pytorch_lightning as pl

# import trainer class, which orchestrates our model training
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# import our model class, to be trained
from IMDBClassifier import ImdbClassifier, ImdbDataModule

# This is the main method, to be run when train.py is invoked
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=0) # used to support multi-GPU or CPU training

    # Data, model, and output directories. Passed by sagemaker with default to os env variables
    parser.add_argument('-o','--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('-m','--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('-tr','--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('-te','--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args, _ = parser.parse_known_args()
    #print(args)

    #tb_logger = TensorBoardLogger(save_dir="tensorboard", name="my_model")
    tb_logger = TensorBoardLogger(save_dir=args.output_data_dir+"/tensorboard",name="imdb_model")

    # Now we have all parameters and hyperparameters available and we need to match them with sagemaker 
    # structure. default_root_dir is set to out_put_data_dir to retrieve from training instances all the 
    # checkpoint and intermediary data produced by lightning
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    

    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    
    model = ImdbClassifier(**vars(args))

    # ------------------------
    # 2 CALLBACKS of MODEL
    # ------------------------

    # callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode='min',
        strict=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
     monitor='val_loss',
     #dirpath='my/path/',
     filename='imdb-classfiy-epoch{epoch:02d}-val_loss{val_loss:.2f}',
     auto_insert_metric_name=False
    )

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args,
        callbacks=[early_stop,lr_monitor, checkpoint_callback]
    )    

    seed_everything(42, workers=True)
    imdb_dm = ImdbDataModule()

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(model,imdb_dm)
    #trainer.validate()
    trainer.test()


    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    # Save the parameters used to construct the model - if you want to save model parameters
    
    