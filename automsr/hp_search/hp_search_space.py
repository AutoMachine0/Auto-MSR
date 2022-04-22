from hyperopt import hp

HP_SEARCH_SPACE = {"learning_rate_decay": hp.choice("learning_rate_decay", [0, 1e-3, 1e-4, 1e-5]),
                   "learning_rate": hp.choice("learning_rate", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
                   "mini_batch": hp.choice("mini_batch", [5, 10, 15, 20, 25, 30]),
                   "embedding_dim": hp.choice("embedding_dim", [8, 16, 32, 64, 128, 256])}

HP_SEARCH_SPACE_Mapping = {"learning_rate_decay": [0, 1e-3, 1e-4, 1e-5],
                           "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                           "mini_batch": [5, 10, 15, 20, 25, 30],
                           "embedding_dim": [8, 16, 32, 64, 128, 256]}