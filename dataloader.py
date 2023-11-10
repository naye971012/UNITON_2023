from glob import glob
import os
from dataset import SegDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def get_loaders(configs):
    """
    get train/valid/test loader
    """
    data_directory = configs.DATA_PATH
    train_image_paths = sorted(glob(os.path.join(data_directory, 'train', 'images', '*.png')))
    train_mask_paths = sorted(glob(os.path.join(data_directory, 'train', 'masks', '*.png')))
    test_image_paths = sorted(glob(os.path.join(data_directory, 'test', 'images', '*.png')))

    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(train_image_paths, 
                                                                                            train_mask_paths, 
                                                                                            test_size=configs.VALI_SIZE, 
                                                                                            random_state=configs.SEED)

    train_dataset = SegDataset(train_image_paths, train_mask_paths, resize=configs.RESIZE, train_transform_name=configs.train_transform)
    val_dataset = SegDataset(val_image_paths, val_mask_paths, mode='val', resize=configs.RESIZE)
    test_dataset = SegDataset(test_image_paths, mode='test', resize=configs.RESIZE)

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size , shuffle=True, num_workers=configs.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size , shuffle=False, num_workers=configs.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size , shuffle=False, num_workers=configs.NUM_WORKERS)

    return train_loader, val_loader, test_loader