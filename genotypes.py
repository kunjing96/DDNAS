N = 6

DARTS = \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * N + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * N + \
[[(('max_pool_3x3', 0), ('max_pool_3x3', 1)), (('skip_connect', 2), ('max_pool_3x3', 1)), (('max_pool_3x3', 0), ('skip_connect', 2)), (('skip_connect', 2), ('max_pool_3x3', 1))]] + \
[[(('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_3x3', 1), ('skip_connect', 0)), (('skip_connect', 0), ('dil_sepc_3x3', 2))]] * N

GDAS = \
[[(('skip_connect', 0), ('skip_connect', 1)), (('skip_connect', 0), ('dua_sepc_5x5', 2)), (('dua_sepc_3x3', 3), ('skip_connect', 0)), (('dua_sepc_5x5', 4), ('dua_sepc_3x3', 3))]] * N + \
[[(('dua_sepc_5x5', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_5x5', 2), ('dua_sepc_5x5', 1)), (('dil_sepc_5x5', 2), ('dua_sepc_3x3', 1)), (('dua_sepc_5x5', 0), ('dua_sepc_5x5', 1))]] + \
[[(('skip_connect', 0), ('skip_connect', 1)), (('skip_connect', 0), ('dua_sepc_5x5', 2)), (('dua_sepc_3x3', 3), ('skip_connect', 0)), (('dua_sepc_5x5', 4), ('dua_sepc_3x3', 3))]] * N + \
[[(('dua_sepc_5x5', 0), ('dua_sepc_3x3', 1)), (('dua_sepc_5x5', 2), ('dua_sepc_5x5', 1)), (('dil_sepc_5x5', 2), ('dua_sepc_3x3', 1)), (('dua_sepc_5x5', 0), ('dua_sepc_5x5', 1))]] + \
[[(('skip_connect', 0), ('skip_connect', 1)), (('skip_connect', 0), ('dua_sepc_5x5', 2)), (('dua_sepc_3x3', 3), ('skip_connect', 0)), (('dua_sepc_5x5', 4), ('dua_sepc_3x3', 3))]] * N

GENOTYPES = {'DARTS': DARTS,
             'GDAS' : GDAS}