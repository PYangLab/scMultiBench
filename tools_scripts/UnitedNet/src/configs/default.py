default_config = {
    "train_batch_size": 512,
    "finetune_batch_size": 5000,
    "transfer_batch_size": 512,
    "train_epochs": 50,
    "finetune_epochs": 50,
    "transfer_epochs": 50,
    "train_task": "supervised_group_identification",
    "finetune_task": "supervised_group_identification",
    "transfer_task": "supervised_group_identification",
    "train_loss_weight": None,
    "finetune_loss_weight": None,
    "transfer_loss_weight": None,
    "lr": 0.01,
    "checkpoint": 1,
    "n_head": 1,
    "encoders": [
        {
            "input": 4163, #13634,
            "hiddens": [64, 64],
            "output": 64,
            "use_biases": [False, False, False],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "relu"],
            "use_batch_norms": [True, True, True],
            "use_layer_norms": [False, False, False],
            "is_binary_input": True,
        },
        {
            "input":8961, # 4000,
            "hiddens": [64, 64],
            "output": 64,
            "use_biases": [False, False, False],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "relu"],
            "use_batch_norms": [False, False, False],
            "use_layer_norms": [False, False, False],
            "is_binary_input": False,
        },
    ],
    "decoders": [
        {
            "input": 64,
            "hiddens": [64, 64],
            "output": 4163, #13634,
            "use_biases": [False, False, False],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", "sigmoid"],
            "use_batch_norms": [False, False, False],
            "use_layer_norms": [False, False, False],
        },
        {
            "input": 64,
            "hiddens": [64, 64],
            "output": 8961,  #4000,
            "use_biases": [False, False, False],
            "dropouts": [0, 0, 0],
            "activations": ["relu", "relu", None],
            "use_batch_norms": [False, False, False],
            "use_layer_norms": [False, False, False],
        },
    ],
    "discriminators": [
        {
            "input": 4163, #13634,
            "hiddens": [64],
            "output": 1,
            "use_biases": [False, False],
            "dropouts": [0, 0],
            "activations": ["relu", "sigmoid"],
            "use_batch_norms": [False, False],
            "use_layer_norms": [False, False],
        },
        {
            "input": 8961,  #4000,
            "hiddens": [64],
            "output": 1,
            "use_biases": [False, False],
            "dropouts": [0, 0],
            "activations": ["relu", "sigmoid"],
            "use_batch_norms": [False, False],
            "use_layer_norms": [False, False],
        },
    ],
    "projectors": {
        "input": 64,
        "hiddens": [],
        "output": 100,
        "use_biases": [False],
        "dropouts": [0],
        "activations": [None],
        "use_batch_norms": [False],
        "use_layer_norms": [False],
    },
    "clusters": {
        "input": 100,
        "hiddens": [],
        "output": 22,
        "use_biases": [False],
        "dropouts": [0],
        "activations": [None],
        "use_batch_norms": [False],
        "use_layer_norms": [False],
    },

}
