import torch
import torch.nn as nn
from operations import OPS
from copy import deepcopy


class MixedOp(nn.Module):

  def __init__(self, space, C, stride, affine, track_running_stats):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in space:
      op = OPS[primitive](C, C, stride, affine, track_running_stats)
      self._ops.append(op)

  def reset_parameters(self):
    for op in self._ops:
      op.reset_parameters()

  def forward(self, x, weights, index):
    return self._ops[index](x) * weights[index]


class NASCell(nn.Module):

  def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
    super(NASCell, self).__init__()
    self.reduction = reduction
    self.op_names  = deepcopy(space)
    if reduction_prev: self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
    else             : self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
    self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
    self._steps = steps
    self._multiplier = multiplier

    self.ops = nn.ModuleDict()
    for i in range(self._steps):
      for j in range(2+i):
        node_str = '{:}<-{:}'.format(i+2, j)  # indicate the edge from node-(j) to node-(i+2)
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(space, C, stride, affine, track_running_stats)
        self.ops[ node_str ] = op
    self.edge_keys  = sorted(list(self.ops.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.ops)

    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self.num_edges, len(space)) )
    self.geno = [ ((None, -1), (None, -1)) ] * self._steps

  @property
  def multiplier(self):
    return self._multiplier

  def set_geno(self, geno):
    assert len(geno) == self._steps, 'The length of genotype of a cell must be {:}.'.format(self._steps)
    for edges in geno:
      assert len(edges) == 2, 'The #edges of a node must be 2.'
      assert (edges[0][0] is None and edges[0][1] == -1) or (edges[0][0] is not None and edges[0][1] != -1)
      assert (edges[1][0] is None and edges[1][1] == -1) or (edges[1][0] is not None and edges[1][1] != -1)
    self.geno = geno

  def get_geno(self):
    def _parse(weights):
      geno = []
      for i in range(self._steps):
        op0, pre0 = geno[i][0][0], geno[i][0][1]
        op1, pre1 = geno[i][1][0], geno[i][1][1]
        if pre0 == -1 or pre1 == -1:
          pres = list(range(2+i))
          if pre0 != -1: pres.remove(pre0)
          if pre1 != -1: pres.remove(pre1)
          edges = []
          for j in pres:
            node_str = '{:}<-{:}'.format(i+2, j)
            ws = weights[ self.edge2index[node_str] ]
            for k, op_name in enumerate(self.op_names):
              if op_name == 'none': continue
              edges.append( (op_name, j, ws[k]) )
          edges = sorted(edges, key=lambda x: -x[-1])
          if pre0 == -1 and pre1 == -1:
            selected_edge_0 = (edges[0][0], edges[0][1])
            for edge in edges[1:]:
              if edge[1] != edges[0][1]:
                selected_edge_1 = (edge[0], edge[1])
                break
          else:
            if   pre0 != -1: selected_edge_0, selected_edge_1 = (op0, pre0), (edges[0][0], edges[0][1])
            elif pre1 != -1: selected_edge_0, selected_edge_1 = (edges[0][0], edges[0][1]), (op1, pre1)
        else: selected_edge_0, selected_edge_1 = (op0, pre0), (op1, pre1)
        geno.append( (selected_edge_0, selected_edge_1) )
      return geno

    with torch.no_grad():
      geno = _parse(torch.softmax(self.arch_parameters, dim=-1).cpu().numpy())
    return geno

  def reset_parameters(self):
    self.arch_parameters = 1e-3*nn.init.normal_(self.arch_parameters)
    for i in range(self._steps):
      op0, pre0 = geno[i][0][0], geno[i][0][1]
      op1, pre1 = geno[i][1][0], geno[i][1][1]
      if pre0 == -1 or pre1 == -1:
        pres = list(range(2+i))
        if pre0 != -1: pres.remove(pre0)
        if pre1 != -1: pres.remove(pre1)
        for j in pres:
          node_str = '{:}<-{:}'.format(i+2, j)
          self.ops[ node_str ].reset_parameters()

  def get_alpha(self):
    return self.arch_parameters

  def show_alpha(self):
    with torch.no_grad():
      str = '{:}'.format( nn.functional.softmax(self.arch_parameters, dim=-1).cpu() )
    return str

  def forward(self, s0, s1, tau): # self.geno
    def get_gumbel_prob(xins, tau):
      while True:
        gumbels = -torch.empty_like(xins).exponential_().log()
        logits  = (xins.log_softmax(dim=1) + gumbels) / tau
        probs   = nn.functional.softmax(logits, dim=1)
        index   = probs.max(-1, keepdim=True)[1]
        one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
          continue
        else: break
      return hardwts, index

    weightss, indexs = get_gumbel_prob(self.arch_parameters, tau)

    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      op0, pre0 = geno[i][0][0], geno[i][0][1]
      op1, pre1 = geno[i][1][0], geno[i][1][1]
      no_mixed_op_edges = {}
      if pre0 != -1: no_mixed_op_edges.update({'{:}<-{:}'.format(i+2, pre0): op0})
      if pre1 != -1: no_mixed_op_edges.update({'{:}<-{:}'.format(i+2, pre1): op1})
      clist = []
      for j, h in enumerate(states):
        node_str = '{:}<-{:}'.format(i+2, j)
        op = self.ops[ node_str ]
        if node_str in no_mixed_op_edges.keys():
          index = self.op_names.index( no_mixed_op_edges[node_str] )
          weights = torch.zeros_like(self.arch_parameters[0])
          weights[index] = 1
        else:
          if len(no_mixed_op_edges) < 2:
            weights = weightss[ self.edge2index[node_str] ]
            index   = indexs[ self.edge2index[node_str] ].item()
          else: continue
        clist.append( op(h, weights, index) )

      states.append( sum(clist) )

    return torch.cat(states[-self._multiplier:], dim=1)

  def extra_repr(self):
    return ('{name}(steps={_steps}, multiplier={_multiplier}, reduction={reduction})'.format(name=self.__class__.__name__, **self.__dict__))


class NASNetwork(nn.Module):

  def __init__(self, C, N, steps, multiplier, stem_multiplier, num_classes, search_space, affine, track_running_stats):
    super(NASNetwork, self).__init__()
    self._C        = C
    self._layerN   = N
    self._steps    = steps
    self._multiplier = multiplier
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C*stem_multiplier, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C*stem_multiplier))
  
    # config for each layer
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * (N-1) + [C*4 ] + [C*4  ] * (N-1)
    layer_reductions = [False] * N + [True] + [False] * (N-1) + [True] + [False] * (N-1)

    num_edge, edge2index = None, None
    C_prev_prev, C_prev, C_curr, reduction_prev = C*stem_multiplier, C*stem_multiplier, C, False

    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = NASCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
      if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
      else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev_prev, C_prev, reduction_prev = C_prev, multiplier*C_curr, reduction
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self.tau = 10

  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

  def set_genos(self, genos):
    assert len(genos) == self._Layer, 'The length of network genotypes must be {:}.'.format(self._Layer)
    for (cell, geno) in zip(self.cells, genos):
      cell.set_geno(geno)

  def get_genos(self):
    return [cell.get_geno() for cell in self.cells]

  def reset_parameters(self):
    for cell in self.cells:
      cell.reset_parameters()

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self):
    return [cell.get_alpha() for cell in self.cells]

  def show_alphas(self):
    str = ''
    for i, cell in enumerate(self.cells):
      str += '\n' if i != 0 else ''
      str += 'cell{:}-arch-parameters :\n'
      str += cell.show_alpha()
    return str

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    s0 = s1 = self.stem(inputs)
    for cell in self.cells:
      s0, s1 = s1, cell(s0, s1, self.tau)
    out = self.lastact(s1)
    out = self.global_pooling(out)
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits
