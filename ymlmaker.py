from fairchem.core.common.tutorial_utils import generate_yml_config
checkpoint = './gnoc_oc22_oc20_all_s2ef.pt'

yml = generate_yml_config(checkpoint, 'config.yml',
                delete=['slurm', 'cmd', 'logger', 'task', 'model_attributes',
                        'optim.loss_force',
                        'optim.load_balancing',
                        'dataset', 'test_dataset', 'val_dataset'],
                update={'gpus': 1,
                        'task.dataset': 'lmdb',
                        'optim.eval_every': 10,
                        'optim.max_epochs': 1,
                        'optim.batch_size': 4,
                        'logger': 'tensorboard',
                        # Train data
                        'dataset.train.src': './data/Ag111_train.lmdb',
                        'dataset.train.format': 'lmdb',
                        'dataset.train.a2g_args.r_energy': True,
                        'dataset.train.a2g_args.r_forces': True,
                        # Test data - prediction only so no regression
                        'dataset.test.src': './data/Ag111_test.lmdb',
                        'dataset.test.format': 'lmdb',
                        'dataset.test.a2g_args.r_energy': False,
                        'dataset.test.a2g_args.r_forces': False,
                        # val data
                        'dataset.val.src': './data/Ag111_val.lmdb',
                        'dataset.val.format': 'lmdb',
                        'dataset.val.a2g_args.r_energy': True,
                        'dataset.val.a2g_args.r_forces': True,
                        })

