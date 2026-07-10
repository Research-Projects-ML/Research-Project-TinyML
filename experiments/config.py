import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'teachers')
SWEEP_DIR = os.path.join(RESULTS_DIR, 'sweep_results')
PIPELINE_DIR = os.path.join(RESULTS_DIR, 'pipeline_results')
EVAL_DIR = os.path.join(RESULTS_DIR, 'eval_results')
SWEEP_SEED = 0
ORDERING_SEEDS = [0, 1, 2]
PRUNE_RATIO = 0.4

HARDWARE_CONFIG = {
    'cortex_m0plus': {
        'board': 'NUCLEO-L073RZ','mcu': 'STM32L073RZT6','core': 'Cortex-M0+',
        'ram_kb': 20,'flash_kb': 192,'clock_mhz': 32,'ei_device': None,'ei_clock_mhz': 40,'latency_scale': 40 / 32,
    },
    'cortex_m4f': {
        'board': 'NUCLEO-F401RE','mcu': 'STM32F401RET6','core': 'Cortex-M4F','ei_device': 'cortex-m4f-80mhz',
        'ram_kb': 96,'flash_kb': 512,'clock_mhz': 84, 'ei_clock_mhz':  80,'latency_scale': 1.0,
    },
    'cortex_m7': {
        'board': 'NUCLEO-F746ZG','mcu': 'STM32F746ZGT6','core': 'Cortex-M7','ei_device': 'cortex-m7-216mhz',
        'ram_kb': 320,'flash_kb': 1024,'clock_mhz': 216,'ei_clock_mhz': 216,'latency_scale': 1.0,
    },
}

BASE_CONFIG = {
    # Pruning
    'prune_ratio': PRUNE_RATIO,
    # Fine-tuning after pruning low LR, model retains learned structure
    'finetune_lr': 1e-4, 'finetune_epochs': 30,
    # Training from scratch (baseline student, no compression)
    'train_lr': 1e-3, 'train_epochs': 100,
    # Shared early stopping patience
    'early_stop_patience': 10,
    # Knowledge distillation
    'kd_learning_rate': 1e-3, 'kd_temperature': 4, 'kd_alpha': 0.7, 'kd_epochs': 50, 'kd_patience': 10,
    # QAT
    'qat_learning_rate': 1e-5,'qat_epochs': 50,'qat_patience': 10,
    # Data
    'batch_size': 128,'data_dir': DATA_DIR,
}

DOMAINS_CONFIG = {
    'timeseries': {
        'num_classes': 6,'input_channels': 9,'teacher_num_blocks': 4,'teacher_num_channels': 64,
        'student_num_blocks': 2,'student_num_channels': 32,'kernel_size': 3,'dropout': 0.1,
        'teacher_path': os.path.join(CHECKPOINT_DIR, 'tcn_teacher.keras'),
    },
    'image': {
        'batch_size': 64,'num_classes': 10,'teacher_base_filters': 32,'student_base_filters': 16,
        'kd_epochs': 80,'kd_patience': 15,'finetune_epochs': 50,
        'teacher_path': os.path.join(CHECKPOINT_DIR, 'resnet8_teacher.keras'),
    },
}

SWEEP_CONFIG = {'prune_ratios': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'seed': SWEEP_SEED,'output_dir': SWEEP_DIR,}

PIPELINES = [
    {'name': 'P_KD_QAT','stages': ['P','KD','QAT']},
    {'name': 'P_QAT_KD','stages': ['P','QAT','KD']},
    {'name': 'KD_P_QAT','stages': ['KD','P','QAT']},
    {'name': 'KD_QAT_P','stages': ['KD','QAT','P']},
    {'name': 'QAT_P_KD','stages': ['QAT','P','KD']},
    {'name': 'QAT_KD_P','stages': ['QAT','KD','P']},
    {'name': 'P_only','stages': ['P']},
    {'name': 'KD_only','stages': ['KD']},
    {'name': 'QAT_only','stages': ['QAT']},
    {'name': 'baseline','stages': []},
]