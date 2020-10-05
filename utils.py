import os, sys, torch, random, PIL, copy, numpy as np


def prepare_seed(rand_seed):
  random.seed(rand_seed)
  np.random.seed(rand_seed)
  torch.manual_seed(rand_seed)
  torch.cuda.manual_seed(rand_seed)
  torch.cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
  args = copy.deepcopy( xargs )
  from log_utils import Logger
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
