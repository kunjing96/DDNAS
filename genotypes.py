GENOTYPES = {'DARTS': DARTS}

DARTS = \
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 6 + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 5 + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 0), ('sep_conv_3x3', 1)), (('sep_conv_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_conv_3x3', 2))]] * 5
