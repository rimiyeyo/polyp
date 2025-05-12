import os

from torch.utils.data import DataLoader, Dataset

from data import (
    x_test_filenames,
    x_train_filenames,
    y_test_filenames,
    y_train_filenames,
)
from utils import _process_pathnames


class SegDataset(Dataset):
	def __init__(self, 
				filenames, 
				labels, 
				# preproc_fn=functools.partial(_augment), 
				is_train=True):
		assert len(filenames) == len(labels), "파일 수와 레이블 수가 일치해야 합니다"
		self.filenames = filenames
		self.labels = labels
		# self.preproc_fn = preproc_fn
		self.is_train = is_train
    
	def __len__(self):
		return len(self.filenames)
  
	def __getitem__(self, idx):
		img, label = _process_pathnames(self.filenames[idx], self.labels[idx])
		 
		# if self.is_train:
		# 	img, label = self.preproc_fn(img, label)
		return img, label

train_dataset = SegDataset(x_train_filenames, y_train_filenames)
test_dataset = SegDataset(x_test_filenames, y_test_filenames, is_train=False)

train_dataloader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              pin_memory=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True)
