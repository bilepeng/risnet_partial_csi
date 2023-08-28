import numpy as np
import torch
import copy


# RT channel model
params2users = {'lr': 8e-4,
                'epoch': 50,
                'num_users': 2,
                'num_ris_antennas': 1296,
                'iter_wmmse': 1100,
                'epoch_per_iter_wmmse': 1,
                'entropy_history_length': 5,
                'alphas': [0.5, 0.5],
                'saving_frequency': 100,
                'wmmse_saving_frequency': 100,
                'batch_size': 512,
                'permutation_invariant': True,
                'results_path': 'results/',
                'tsnr': 1e12,
                "frequencies": np.linspace(1e6, 1e6 + 100, 10),
                'quantile2keep': 0.6,
                "phase_shift": "continuous",
                "discrete_phases": torch.tensor([0, np.pi])[None, None, :],
                'mean_ris': 7.75e-5,
                'std_ris': 2.49e-5,
                'mean_direct': 1.16e-5,
                'std_direct': 6.46e-5,
                'ris_shape': (36, 36),
                'channel_tx_ris_original_shape': (36, 36, 9),
                # width, height of RIS and Tx antennas. Do not change this!
                'channel_ris_rx_original_shape': (20480, 36, 36),  # samples, width, height of RIS and users
                'channel_ris_rx_original_shape_testing': (1024, 36, 36),  # samples, width, height of RIS and users
                'n_tx_antennas': 9,
                'los': True,
                'precoding': 'wmmse',
                # Debug
                'channel_direct_path': 'data/channels_direct_training.pt',
                'channel_tx_ris_path': 'data/channel_tx_ris.pt',
                'channel_ris_rx_path': 'data/channels_ris_rx_training.pt',
                'location_path': 'data/locations_training.pt',
                'group_definition_path': 'data/group_definition_2users_training_s.npy',
                'channel_direct_testing_path': 'data/channels_direct_testing.pt',
                'channel_ris_rx_testing_path': 'data/channels_ris_rx_testing.pt',
                'location_testing_path': 'data/locations_training.pt',
                'group_definition_testing_path': 'data/group_definition_2users_training_s.npy',
                # 'channel_direct_path': 'data/channels_direct_training_s.pt',
                # 'channel_tx_ris_path': 'data/channel_tx_ris_s.pt',
                # 'channel_ris_rx_path': 'data/channels_ris_rx_training_s.pt',
                'angle_diff_threshold': 0.5,
                'user_distance_threshold': 20,
                'ris_loc': torch.tensor([278.42, 576.97, 2]),
                'trained_mmse_model': None,
                # 'trained_mmse_model': 'results/RISNetPIDiscrete_MMSE_16-05-2022_13-46-01/ris_100000000000.0_(32, 32)_[0.5, 0.5]_4000',
                'channel_estimate_error': 0,
                'discount_long': 0.95,
                'discount_short': 0.4,
                'delta_support': 0.0001,
                "discrete_phase_granularity": np.pi / 2,

                # RISnet setting
                "skip_connection": False,
                "global_info_not_opposite_info_antennas": True,
                "global_info_not_opposite_info_users": False,
                "normalize_phase": False,
                "partial_csi": True,
                "indices_sensors": [4, 13, 22, 31],
                }


params4users = copy.deepcopy(params2users)
params4users["num_users"] = 4
params4users['group_definition_path'] = 'data/group_definition_4users_training.npy'
params4users['group_definition_testing_path'] = 'data/group_definition_4users_testing.npy'
params4users["alphas"] = [0.25, 0.25, 0.25, 0.25]

params = params4users