patch_size = 10

configs_pretrain = {
    "image_size": (1, 300),
    "patch_size1": (1, 3),
    "patch_size2": (1, 5),
    "patch_size3": (1, None),
    "ch_1": 14, 
    "ch_2": 32, 
    "ch_3": None, 
    "tcn_layers": 2,
    "dim_patch": 64,
    "dim_head": 32,
    "heads": 8,
    "depth": 1,
    "channels": 14,
    "num_classes": 8,
    "dropout": .2,
    "emb_dropout": 0,
    "pool": "cls",
    "use_cls_token": True,
    # "sessions": 1,
    "sessions": 5,
    # "subjects": [(2,)],
    "subjects": [(2, 3, 4, 5, 6, 7, 8, 9, 10), (1, 3, 4, 5, 6, 7, 8, 9, 10), (1, 2, 4, 5, 6, 7, 8, 9, 10), (1, 2, 3, 5, 6, 7, 8, 9, 10), (1, 2, 3, 4, 6, 7, 8, 9, 10), (1, 2, 3, 4, 5, 7, 8, 9, 10), (1, 2, 3, 4, 5, 6, 8, 9, 10), (1, 2, 3, 4, 5, 6, 7, 9, 10), (1, 2, 3, 4, 5, 6, 7, 8, 10), (1, 2, 3, 4, 5, 6, 7, 8, 9)],
    "pretrained": None,
    "training_config": [
        {
         "epochs": 100,
         "batch_size": 64,
         "optim": "Adam",
         "optim_hparams": {"lr": 0, "betas": (.9, .999), "weight_decay": 0},
         "lr_scheduler": "CyclicLR",
         "lr_scheduler_hparams": {"base_lr": 1e-7, "max_lr": 1e-3, "step_size_up": 200, "step_size_down": None, "mode": 'triangular', "cycle_momentum": False},
        },
    ]
}

configs_finetune = {
    "image_size": (1, 300),
    "patch_size1": (1, 3),
    "patch_size2": (1, 5),
    "patch_size3": (1, None),
    "ch_1": 14, 
    "ch_2": 32, 
    "ch_3": None, 
    "tcn_layers": 2,
    "dim_patch": 64,
    "dim_head": 32,
    "heads": 8,
    "depth": 1,
    "channels": 14,
    "num_classes": 8,
    "dropout": .2,
    "emb_dropout": 0,
    "pool": "cls",
    "use_cls_token": True,
    "sessions": 5,
    "subjects": [1,2,3,4,5,6,7,8,9,10],
    "pretrained": True,
    "training_config": [
        {
         "epochs": 20,
         "batch_size": 8,
         "optim": "Adam",
         "optim_hparams": {"lr": 1e-4, "betas": (.9, .999), "weight_decay": 0},
         "lr_scheduler": "StepLR",
         "lr_scheduler_hparams": {"gamma": .1, "step_size": 10},
        },
    ]
}


configs_finetune_nopretrain = {
    "image_size": (1, 300),
    "patch_size1": (1, 3),
    "patch_size2": (1, 5),
    "patch_size3": (1, None),
    "ch_1": 14, 
    "ch_2": 32, 
    "ch_3": None, 
    "tcn_layers": 2,
    "dim_patch": 64,
    "dim_head": 32,
    "heads": 8,
    "depth": 1,
    "channels": 14,
    "num_classes": 8,
    "dropout": .2,
    "emb_dropout": 0,
    "pool": "cls",
    "use_cls_token": True,
    "sessions": 5,
    "subjects": [1,2,3,4,5,6,7,8,9,10],
    "pretrained": True,
    "training_config": [
        {
         "epochs": 40,
         "batch_size": 8,
         "optim": "Adam",
         "optim_hparams": {"lr": 1e-4, "betas": (.9, .999), "weight_decay": 0},
         "lr_scheduler": "StepLR",
         "lr_scheduler_hparams": {"gamma": .1, "step_size": 10},
        },
    ]
}


configs_quantization = {
    "image_size": (1, 300),
    "patch_size1": (1, 3),
    "patch_size2": (1, 5),
    "patch_size3": (1, None),
    "ch_1": 14, 
    "ch_2": 32, 
    "ch_3": None, 
    "tcn_layers": 2,
    "dim_patch": 64,
    "dim_head": 32,
    "heads": 8,
    "depth": 1,
    "channels": 14,
    "num_classes": 8,
    "dropout": .2,
    "emb_dropout": 0,
    "pool": "cls",
    "use_cls_token": True,
    "sessions": 5,
    "subjects": [1,2,3,4,5,6,7,8,9,10],
    "pretrained": True,
    "training_config": [
        {
         "epochs": 10,
         "batch_size": 64,
         "optim": "Adam",
         "optim_hparams": {"lr": 5e-5, "betas": (.9, .999), "weight_decay": 0},
         "lr_scheduler": "StepLR",
         "lr_scheduler_hparams": {"gamma": .1, "step_size": 10},
        },
    ]
}
