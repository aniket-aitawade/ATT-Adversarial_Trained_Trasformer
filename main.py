from glob import glob
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import json
import tensorflow as tf
import model
import dataloader
import preprocessing
import utils

config=OmegaConf.load('config.yaml')

# Data Preparation
dataloader.load_data(config)

# Data Preprocessing
print("Data Processing Stage")
preprocessing.load_config(config)

# Load Dataset
with open('dataset.json','r') as file:
    data=json.load(file)

dataset_train=tf.data.Dataset.from_tensor_slices((data['train']['noisy'],data['train']['clean']))
dataset_val=tf.data.Dataset.from_tensor_slices((data['val']['noisy'],data['val']['clean']))
dataset_test=tf.data.Dataset.from_tensor_slices((data['test']['noisy'],data['test']['clean']))
processed_dataset_train = (dataset_train.map(preprocessing.get_spec, num_parallel_calls=tf.data.AUTOTUNE)
                     .padded_batch(config.preprocessing.batch_size)
                     .prefetch(buffer_size=tf.data.AUTOTUNE))
processed_dataset_val = (dataset_val.map(preprocessing.get_spec, num_parallel_calls=tf.data.AUTOTUNE)
                     .padded_batch(config.preprocessing.batch_size)
                     .prefetch(buffer_size=tf.data.AUTOTUNE))
processed_dataset_test = (dataset_test.map(preprocessing.get_spec, num_parallel_calls=tf.data.AUTOTUNE)
                     .padded_batch(config.preprocessing.batch_size)
                     .prefetch(buffer_size=tf.data.AUTOTUNE))

#Model Building
g_opt=tf.keras.optimizers.RMSprop(learning_rate=config.training.learning_rate)
d_opt=tf.keras.optimizers.RMSprop(learning_rate=config.training.learning_rate)

ATT_model=model.ATT(512)
ATT_model.compile(g_opt,g_opt)

# processed_dataset_train=processed_dataset_train.take(10)
# processed_dataset_val=processed_dataset_val.take(3)

history=ATT_model.fit(processed_dataset_train,
                      validation_data=processed_dataset_val,
                      epochs=config.training.epochs,
                      callbacks=[ utils.model_checkpoints_callback(),
                                  utils.tensorboard_callback()])