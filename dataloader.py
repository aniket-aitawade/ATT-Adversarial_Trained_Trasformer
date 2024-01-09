from glob import glob
import os
from omegaconf import OmegaConf
import json

def load_data(config):
    print("Data Preparation Stage")
    # config=OmegaConf.load(config_file)
    audios_train=os.listdir(os.path.join(config.directory,'train','noisy'))
    audios_test=os.listdir(os.path.join(config.directory,'test','noisy'))

    train_set=audios_train[:int(len(audios_train)*config.train_val_split)]
    val_set=audios_train[int(len(audios_train)*config.train_val_split):]

    train_set={'noisy':[os.path.join(config.directory,'train','noisy',i) for i in train_set],
            'clean':[os.path.join(config.directory,'train','clean',i) for i in train_set]}

    val_set={'noisy':[os.path.join(config.directory,'train','noisy',i) for i in val_set],
            'clean':[os.path.join(config.directory,'train','clean',i) for i in val_set],}

    test_set={'noisy':[os.path.join(config.directory,'test','noisy',i) for i in audios_test],
            'clean':[os.path.join(config.directory,'test','clean',i) for i in audios_test],}

    dataset={'train':train_set,'val':val_set,'test':test_set}
    with open('dataset.json','w') as file:
        json.dump(dataset,file)

if __name__=="__main__":
    load_data()