import sys, time, math, random, argparse
from copy import deepcopy
import torch
from data import get_datasets, get_nas_search_loaders
from utils import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, obtain_accuracy, AverageMeter, time_string, convert_secs2time, load_config, dict2config, get_optim_scheduler, compute_num_unpruned_edges
from flop_benchmark import get_model_infos
from operations import SearchSpaceNames


def search_func_v1(xloader, network, criterion, auxiliary, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (inputs, targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights and the architecture-weight
    w_optimizer.zero_grad()
    a_optimizer.zero_grad()
    logits, logits_aux = network(inputs)
    loss = criterion(logits, targets)
    if logits_aux is not None:
      loss_aux = criterion(logits_aux, targets)
      loss = loss + auxiliary*loss_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    a_optimizer.step()
    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1, top5=top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return losses.avg, top1.avg, top5.avg


def search_func_v2(xloader, network, criterion, auxiliary, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    logits, logits_aux = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    if logits_aux is not None:
      base_loss_aux = criterion(logits_aux, base_targets)
      base_loss = base_loss + auxiliary*base_loss_aux
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    a_optimizer.zero_grad()
    logits, logits_aux = network(arch_inputs)
    arch_loss = criterion(logits, arch_targets)
    if logits_aux is not None:
      arch_loss_aux = criterion(logits_aux, arch_targets)
      arch_loss = arch_loss + auxiliary*arch_loss_aux
    arch_loss.backward()
    a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
    arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def train_func(xloader, network, criterion, auxiliary, scheduler, optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (inputs, targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    optimizer.zero_grad()
    logits, logits_aux = network(inputs)
    loss = criterion(logits, targets)
    if logits_aux is not None:
      loss_aux = criterion(logits_aux, targets)
      loss = loss + auxiliary*loss_aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    optimizer.step()
    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*TRAIN* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1, top5=top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return losses.avg, top1.avg, top5.avg


def test_func(xloader, network, criterion, auxiliary, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.eval()
  end = time.time()
  for step, (inputs, targets) in enumerate(xloader):
    targets = targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    logits, logits_aux = network(inputs)
    loss = criterion(logits, targets)
    if logits_aux is not None:
      loss_aux = criterion(logits_aux, targets)
      loss = loss + auxiliary*loss_aux
    # record
    prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
    losses.update(loss.item(),  inputs.size(0))
    top1.update  (prec1.item(), inputs.size(0))
    top5.update  (prec5.item(), inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*TEST* ' + time_string() + ' [{:03d}/{:03d}]'.format(step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1, top5=top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return losses.avg, top1.avg, top5.avg


def bn_calibration(m, cumulative_bn_stats=True):
  if isinstance(m, torch.nn.BatchNorm2d):
    m.reset_running_stats()
    m.train()
    #if cumulative_bn_stats:
    #  m.momentum = None


def bn_calibration_v1(xloader, network, search_model, max_iter):
  network.train()
  search_model.apply(bn_calibration)
  for step, (inputs, targets) in enumerate(xloader):
    if step >= max_iter: break
    targets = targets.cuda(non_blocking=True)
    _, _ = network(inputs)


def bn_calibration_v2(xloader, network, search_model, max_iter):
  network.train()
  search_model.apply(bn_calibration)
  for step, (inputs, targets, _, _) in enumerate(xloader):
    if step >= max_iter: break
    targets = targets.cuda(non_blocking=True)
    _, _ = network(inputs)


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, test_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, xargs.cutout)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, train_loader, test_loader = get_nas_search_loaders(train_data, test_data, xargs.dataset, config.batch_size, xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Train-Loader-Num={:}, Test-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(train_loader), len(test_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = SearchSpaceNames[xargs.search_space_name]
  model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space' : search_space, 'affine' : False, 'track_running_stats': True}, None)
  if xargs.dataset == 'cifar10' or xargs.dataset == 'cifar100':
    from models import NASNetworkCIFAR as NASNetwork
  else:
    from models import NASNetworkImageNet as NASNetwork
  search_model = NASNetwork(model_config.C, model_config.N, model_config.steps, model_config.multiplier, model_config.stem_multiplier, model_config.num_classes, model_config.drop_prob, model_config.drop_path_prob, model_config.space, model_config.affine, model_config.track_running_stats, config.auxiliary, config.which_forward)
  logger.log('search-model :\n{:}'.format(search_model))
  logger.log('model-config : {:}'.format(model_config))
  
  w_optimizer, w_scheduler, criterion, criterion_smooth = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  logger.log('criterion_smooth : {:}'.format(criterion_smooth))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  flop, param = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  if criterion_smooth is not None: criterion_smooth = criterion_smooth.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    pruned_genotypes = checkpoint['pruned_genotypes']
    alphas      = checkpoint['alphas']
    alpha_probs = checkpoint['alpha_probs']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes, pruned_genotypes, alphas, alpha_probs = 0, {'best': -1}, {-1: search_model.genos}, {-1: search_model.get_genos()}, {-1: search_model.show_alphas(softmax=False)}, {-1: search_model.show_alphas(softmax=True)}

  # start training
  from genotypes import GENOTYPES
  start_time, search_time, epoch_time, warmup, total_epoch, gamma, genos, total_edges, num_unpruned_edges_1, num_unpruned_edges_2, has_arch_changed = time.time(), AverageMeter(), AverageMeter(), config.warmup, (config.warmup+config.epochs)*2, config.gamma, GENOTYPES[xargs.init_genos], (model_config.N*3+2)*model_config.steps*2, 0, 0, False
  for epoch in range(start_epoch, total_epoch):
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    # update lr
    w_scheduler.update(epoch%(total_epoch//2), 0.0)
    logger.log('\n[The {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))
    # update drop_path_prob
    if hasattr(search_model, 'update_drop_path'): search_model.update_drop_path(model_config.drop_path_prob * (epoch%(total_epoch//2)) / (total_epoch//2-1))
    # set initial genos
    if epoch == 0:
      search_model.set_genos(genos)
      logger.log('[{:}] set new genos {:}.'.format(epoch_str, xargs.init_genos))

    if epoch >= total_epoch//2:
      # train
      train_loss, train_top1, train_top5 = train_func(train_loader, network, criterion if criterion_smooth is None else criterion_smooth, config.auxiliary, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
      search_time.update(time.time() - start_time)
      logger.log('[{:}] training : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s.'.format(epoch_str, train_loss, train_top1, train_top5, search_time.sum))

      # valid
      with torch.no_grad():
        test_loss, test_top1, test_top5 = test_func(test_loader, network, criterion, config.auxiliary, xargs.print_freq, logger)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%.'.format(epoch_str, test_loss , test_top1 , test_top5 ))
    else:
      # update tau
      search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch//2-1) )
      logger.log('[{:}] tau={:}.'.format(epoch_str, search_model.get_tau()))

      # update birth rate and bear edges
      gamma_birth_rate = gamma * (epoch / (total_epoch//2-1)**2)
      birth_rate = gamma_birth_rate * (1 + math.cos(math.pi * epoch / (total_epoch//2-1))) / 2 if epoch != total_epoch//2 - 1 else 0
      search_model.birth(birth_rate)
      num_unpruned_edges_1 = compute_num_unpruned_edges(search_model.genos)
      has_arch_changed = has_arch_changed or (num_unpruned_edges_1 != num_unpruned_edges_2)
      logger.log('[{:}] birth_rate={:}, num_unpruned_edges={:}.'.format(epoch_str, birth_rate, num_unpruned_edges_1))

      if has_arch_changed:
        # clear grad
        for param in search_model.parameters():
          param.grad = None
        # clear grad stat
        from collections import defaultdict
        for group in w_optimizer.param_groups:
          for p in group['params']:
            w_optimizer.state[p] = defaultdict(dict)
        for group in a_optimizer.param_groups:
          for p in group['params']:
            a_optimizer.state[p] = defaultdict(dict)
        # bn recalibration
        if config.bn_recalibration:
          if config.optimization == 'one-level':
            with torch.no_grad():
              bn_calibration_v1(train_loader, network, search_model, config.bn_calibration_step)
          elif config.optimization == 'bi-level':
            with torch.no_grad():
              bn_calibration_v2(search_loader, network, search_model, config.bn_calibration_step)
          else:
            raise ValueError('No {:} optimization implementation!'.format(config.optimization))

      has_arch_changed = False

      #search
      if config.optimization == 'one-level':
        search_loss, search_top1, search_top5 = search_func_v1(train_loader, network, criterion if criterion_smooth is None else criterion_smooth, config.auxiliary, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger)
      elif config.optimization == 'bi-level':
        search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5 = search_func_v2(search_loader, network, criterion if criterion_smooth is None else criterion_smooth, config.auxiliary, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs.print_freq, logger)
      else:
        raise ValueError('No {:} optimization implementation!'.format(config.optimization))
      search_time.update(time.time() - start_time)
      if config.optimization == 'one-level':
        logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s.'.format(epoch_str, search_loss, search_top1, search_top5, search_time.sum))
      elif config.optimization == 'bi-level':
        logger.log('[{:}] searching : base loss={:.2f}, base accuracy@1={:.2f}%, base accuracy@5={:.2f}%, arch loss={:.2f}, arch accuracy@1={:.2f}%, arch accuracy@5={:.2f}%, time-cost={:.1f} s.'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5, search_time.sum))
      else:
        raise ValueError('No {:} optimization implementation!'.format(config.optimization))

      # valid
      with torch.no_grad():
        test_loss, test_top1, test_top5 = test_func(test_loader, network, criterion, config.auxiliary, xargs.print_freq, logger)
      logger.log('[{:}] evaluate : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%.'.format(epoch_str, test_loss , test_top1 , test_top5 ))

      # update prune rate and prune edges
      gamma_prune_rate = 100 * (birth_rate * (total_edges-num_unpruned_edges_2) + num_unpruned_edges_2) / (gamma * birth_rate * (total_edges-num_unpruned_edges_2) + num_unpruned_edges_2 + 1e-6) * (epoch / (total_epoch//2-1)**2)
      prune_rate = gamma_prune_rate * (1 + math.cos(math.pi * epoch / (total_epoch//2-1) + math.pi)) / 2 if epoch != total_epoch//2 - 1 else 1
      search_model.prune(prune_rate)
      num_unpruned_edges_2 = compute_num_unpruned_edges(search_model.genos)
      has_arch_changed = has_arch_changed or (num_unpruned_edges_1 != num_unpruned_edges_2)
      logger.log('[{:}] prune_rate={:}, num_unpruned_edges={:}.'.format(epoch_str, prune_rate, num_unpruned_edges_2))

    genotypes[epoch]        = search_model.genos
    pruned_genotypes[epoch] = search_model.get_genos()
    alphas[epoch]           = search_model.show_alphas(softmax=False)
    alpha_probs[epoch]      = search_model.show_alphas(softmax=True)

    # check the best accuracy
    valid_accuracies[epoch] = test_top1
    if test_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = test_top1
      genotypes['best']        = search_model.genos
      pruned_genotypes['best'] = search_model.get_genos()
      alphas['best']           = search_model.show_alphas(softmax=False)
      alpha_probs['best']      = search_model.show_alphas(softmax=True)
      find_best = True
    else: find_best = False

    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'genotypes'   : genotypes,
                'pruned_genotypes' : pruned_genotypes,
                'alphas'      : alphas,
                'alpha_probs' : alpha_probs,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest test accuracy : {:.2f}%.'.format(epoch_str, test_top1))
      copy_checkpoint(model_base_path, model_best_path, logger)
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

    torch.cuda.empty_cache()

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('DDNAS : run {:} epochs, cost {:.1f} s, last-genos is {:}, last-pruned-genos is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1], pruned_genotypes[total_epoch-1]))
  logger.close()
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser("DDNAS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset.')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between cifar10/100 and imagenet.')
  parser.add_argument('--cutout',             type=int,   help='Cutout length.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='Learning rate for arch encoding.')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='Weight decay for arch encoding.')
  parser.add_argument('--tau_min',            type=float,               help='The minimum tau for Gumbel.')
  parser.add_argument('--tau_max',            type=float,               help='The maximum tau for Gumbel.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='Number of data loading workers (default: 2).')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--print_freq',         type=int,   help='Print frequency (default: 200).')
  parser.add_argument('--rand_seed',          type=int,   help='Manual seed.')
  parser.add_argument('--init_genos',         type=str,   help='Initial genotypes.')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
