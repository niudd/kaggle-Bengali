import random
import os
import numpy as np
import torch
import shutil
import logging


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #tf.set_random_seed(seed)

def set_n_get_device(device_id, data_device_id="cuda:0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id#"0"#"0, 1, 2, 3, 4, 5"
    device = torch.device(data_device_id if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #torch.set_num_threads(20)
    return device

##----------------------------------------------------------------------------------------------------------------
def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    if 'state_dict' not in state.keys() or 'optim_dict' not in state.keys() or \
    'epoch' not in state.keys() or 'metrics' not in state.keys():
        raise ValueError('save_checkpoint: must at least contains state_dict, optim_dict, metrics, epoch')
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        #print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
        #print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return model, optimizer#checkpoint

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
##----------------------------------------------------------------------------------------------------------------

#### transform raw parquet into feather ####
#train_image = pd.read_parquet('../data/raw/train_image_data_%d.parquet'%i, engine='fastparquet')
#train_image.to_feather('../data/processed/train_image_data_%d.feather'%i)


#plt.imshow(train_images_arr[1], cmap='gray')

# ##load train & vaid split, if not exist, process and save
# if not os.path.isfile('../data/processed/train-test-split-seed%d.pkl'%SEED):
#     train_img, valid_img, train_label, valid_label = train_test_split(train_images_arr, train_label_df, 
#                                                                       test_size=0.2, 
#                                                                       stratify=None, 
#                                                                       random_state=SEED)
#     with open('../data/processed/train-test-split-seed%d.pkl'%SEED, 'wb') as f:
#         pickle.dump([train_img, valid_img, train_label, valid_label], f, protocol=4)
#     print('Processed train-test-split, and Saved')
# else:
#     print('Loading train-test-split')
#     with open('../data/processed/train-test-split-seed%d.pkl'%SEED, 'rb') as f:
#         train_img, valid_img, train_label, valid_label = pickle.load(f)


