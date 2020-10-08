import os, sys, torch, random, PIL, copy, json, numpy as np
from collections import namedtuple


def prepare_seed(rand_seed):
  random.seed(rand_seed)
  np.random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
  args = copy.deepcopy( xargs )
  from logger import Logger
  logger = Logger(args.save_dir, args.rand_seed)
  logger.log('Main Function with logger : {:}'.format(logger))
  logger.log('Arguments : -------------------------------')
  for name, value in args._get_kwargs():
    logger.log('{:16} : {:}'.format(name, value))
  logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
  logger.log("Pillow  Version  : {:}".format(PIL.__version__))
  logger.log("PyTorch Version  : {:}".format(torch.__version__))
  logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
  logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
  logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
  logger.log("CUDA_VISIBLE_DEVICES : {:}".format(os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))
  return logger


def get_machine_info():
  info = "Python  Version  : {:}".format(sys.version.replace('\n', ' '))
  info+= "\nPillow  Version  : {:}".format(PIL.__version__)
  info+= "\nPyTorch Version  : {:}".format(torch.__version__)
  info+= "\ncuDNN   Version  : {:}".format(torch.backends.cudnn.version())
  info+= "\nCUDA available   : {:}".format(torch.cuda.is_available())
  info+= "\nCUDA GPU numbers : {:}".format(torch.cuda.device_count())
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    info+= "\nCUDA_VISIBLE_DEVICES={:}".format(os.environ['CUDA_VISIBLE_DEVICES'])
  else:
    info+= "\nDoes not set CUDA_VISIBLE_DEVICES"
  return info


support_types = ('str', 'int', 'bool', 'float', 'none')

def convert_param(original_lists):
  assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
  ctype, value = original_lists[0], original_lists[1]
  assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
  is_list = isinstance(value, list)
  if not is_list: value = [value]
  outs = []
  for x in value:
    if ctype == 'int':
      x = int(x)
    elif ctype == 'str':
      x = str(x)
    elif ctype == 'bool':
      x = bool(int(x))
    elif ctype == 'float':
      x = float(x)
    elif ctype == 'none':
      if x.lower() != 'none':
        raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
      x = None
    else:
      raise TypeError('Does not know this type : {:}'.format(ctype))
    outs.append(x)
  if not is_list: outs = outs[0]
  return outs


def load_config(path, extra, logger):
  path = str(path)
  if hasattr(logger, 'log'): logger.log(path)
  assert os.path.exists(path), 'Can not find {:}'.format(path)
  # Reading data back
  with open(path, 'r') as f:
    data = json.load(f)
  content = { k: convert_param(v) for k,v in data.items()}
  assert extra is None or isinstance(extra, dict), 'invalid type of extra : {:}'.format(extra)
  if isinstance(extra, dict): content = {**content, **extra}
  Arguments = namedtuple('Configure', ' '.join(content.keys()))
  content   = Arguments(**content)
  if hasattr(logger, 'log'): logger.log('{:}'.format(content))
  return content


def dict2config(xdict, logger):
  assert isinstance(xdict, dict), 'invalid type : {:}'.format( type(xdict) )
  Arguments = namedtuple('Configure', ' '.join(xdict.keys()))
  content   = Arguments(**xdict)
  if hasattr(logger, 'log'): logger.log('{:}'.format(content))
  return content


def get_optim_scheduler(parameters, config):
  assert hasattr(config, 'optim') and hasattr(config, 'scheduler') and hasattr(config, 'criterion'), 'config must have optim / scheduler / criterion keys instead of {:}'.format(config)
  if config.optim == 'SGD':
    optim = torch.optim.SGD(parameters, config.LR, momentum=config.momentum, weight_decay=config.decay, nesterov=config.nesterov)
  elif config.optim == 'RMSprop':
    optim = torch.optim.RMSprop(parameters, config.LR, momentum=config.momentum, weight_decay=config.decay)
  else:
    raise ValueError('invalid optim : {:}'.format(config.optim))

  if config.scheduler == 'cos':
    from optimizers import CosineAnnealingLR
    T_max = getattr(config, 'T_max', config.epochs)
    scheduler = CosineAnnealingLR(optim, config.warmup, config.epochs, T_max, config.eta_min)
  elif config.scheduler == 'multistep':
    from optimizers import MultiStepLR
    scheduler = MultiStepLR(optim, config.warmup, config.epochs, config.milestones, config.gammas)
  elif config.scheduler == 'exponential':
    from optimizers import ExponentialLR
    scheduler = ExponentialLR(optim, config.warmup, config.epochs, config.gamma)
  elif config.scheduler == 'linear':
    from optimizers import LinearLR
    scheduler = LinearLR(optim, config.warmup, config.epochs, config.LR, config.LR_min)
  else:
    raise ValueError('invalid scheduler : {:}'.format(config.scheduler))

  if config.criterion == 'Softmax':
    criterion = torch.nn.CrossEntropyLoss()
  elif config.criterion == 'SmoothSoftmax':
    from optimizers import CrossEntropyLabelSmooth
    criterion = CrossEntropyLabelSmooth(config.class_num, config.label_smooth)
  else:
    raise ValueError('invalid criterion : {:}'.format(config.criterion))
  return optim, scheduler, criterion


class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val   = 0.0
    self.avg   = 0.0
    self.sum   = 0.0
    self.count = 0.0
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count    

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))


def convert_secs2time(epoch_time, return_str=False):    
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)  
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if return_str:
    str = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return str
  else:
    return need_hour, need_mins, need_secs


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


def obtain_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


def save_checkpoint(state, filename, logger):
  if osp.isfile(filename):
    if hasattr(logger, 'log'): logger.log('Find {:} exist, delete is at first before saving'.format(filename))
    os.remove(filename)
  torch.save(state, filename)
  assert osp.isfile(filename), 'save filename : {:} failed, which is not found.'.format(filename)
  if hasattr(logger, 'log'): logger.log('save checkpoint into {:}'.format(filename))
  return filename


def copy_checkpoint(src, dst, logger):
  if osp.isfile(dst):
    if hasattr(logger, 'log'): logger.log('Find {:} exist, delete is at first before saving'.format(dst))
    os.remove(dst)
  copyfile(src, dst)
  if hasattr(logger, 'log'): logger.log('copy the file from {:} into {:}'.format(src, dst))


def disturb(original_genos, disturb_rate):
  new_genos = []
  for i in range(len(original_genos)):
    new_geno = []
    for j in range(len(original_genos[i])):
      l, r = original_genos[i][j][0], original_genos[i][j][1]
      if l[0] is not None and l[1] != -1 and np.random.random() < disturb_rate:
        l = (None, -1)
      if r[0] is not None and r[1] != -1 and np.random.random() < disturb_rate:
        r = (None, -1)
      new_geno.append((l, r))
    new_genos.append(new_geno)
  return new_genos
