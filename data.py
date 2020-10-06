import random, torch, numpy as np
import os.path as osp
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms


Dataset2Class = {'cifar10' : 10,
                 'cifar100': 100,
                 'imagenet': 1000}


class SearchDataset(data.Dataset):

  def __init__(self, name, data, train_split, valid_split, check=True):
    self.datasetname = name
    if isinstance(data, (list, tuple)): # new type of SearchDataset
      assert len(data) == 2, 'invalid length: {:}'.format( len(data) )
      self.train_data  = data[0]
      self.valid_data  = data[1]
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      self.mode_str    = 'V2' # new mode 
    else:
      self.mode_str    = 'V1' # old mode 
      self.data        = data
      self.train_split = train_split.copy()
      self.valid_split = valid_split.copy()
      if check:
        intersection = set(train_split).intersection(set(valid_split))
        assert len(intersection) == 0, 'the splitted train and validation sets should have no intersection'
    self.length = len(self.train_split)

  def __repr__(self):
    return ('{name}(name={datasetname}, train={tr_L}, valid={val_L}, version={ver})'.format(name=self.__class__.__name__, datasetname=self.datasetname, tr_L=len(self.train_split), val_L=len(self.valid_split), ver=self.mode_str))

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    assert index >= 0 and index < self.length, 'invalid index = {:}'.format(index)
    train_index = self.train_split[index]
    valid_index = random.choice( self.valid_split )
    if self.mode_str == 'V1':
      train_image, train_label = self.data[train_index]
      valid_image, valid_label = self.data[valid_index]
    elif self.mode_str == 'V2':
      train_image, train_label = self.train_data[train_index]
      valid_image, valid_label = self.valid_data[valid_index]
    else: raise ValueError('invalid mode : {:}'.format(self.mode_str))
    return train_image, train_label, valid_image, valid_label


class Cutout(object):

  def __init__(self, length):
    self.length = length

  def __repr__(self):
    return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


def get_datasets(name, root, cutout):

  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std  = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name == 'imagenet':
    mean = [0.229, 0.224, 0.225]
    std  = [0.229, 0.224, 0.225]
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100':
    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    if cutout > 0: train_transform.transforms.append(Cutout(cutout))
    test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    xshape = (1, 3, 32, 32)
  elif name == 'imagenet':
    train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    xshape = (1, 3, 224, 224)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))

  if name == 'cifar10':
    train_data = dset.CIFAR10 (root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR10 (root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data), len(test_data), 50000, 10000)
  elif name == 'cifar100':
    train_data = dset.CIFAR100(root, train=True , transform=train_transform, download=True)
    test_data  = dset.CIFAR100(root, train=False, transform=test_transform , download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data), len(test_data), 50000, 10000)
  elif name == 'imagenet':
    train_data = dset.ImageFolder(osp.join(root, 'train'), train_transform)
    test_data  = dset.ImageFolder(osp.join(root, 'val'),   test_transform)
    assert len(train_data) == 1281167 and len(test_data) == 50000, 'invalid number of images : {:} & {:} vs {:} & {:}'.format(len(train_data), len(test_data), 1281167, 50000)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))
  
  class_num = Dataset2Class[name]
  return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, test_data, dataset, batch_size, workers):
  if isinstance(batch_size, (list,tuple)):
    batch, test_batch = batch_size
  else:
    batch, test_batch = batch_size, batch_size

  if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'imagenet':
    split = list(range(len(train_data)))
    random.shuffle(split)
    train_split = sorted(split[:len(train_data) // 2])
    valid_split = sorted(split[-(len(train_data) // 2):])
    search_data   = SearchDataset(dataset, train_data, train_split, valid_split)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data , batch_size=batch, shuffle=True , num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data , batch_size=test_batch, shuffle=False , num_workers=workers, pin_memory=True)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))
  return search_loader, train_loader, test_loader
