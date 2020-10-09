DARTS = \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * 6 + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * 5 + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * 5

GENOTYPES = {'DARTS': DARTS}