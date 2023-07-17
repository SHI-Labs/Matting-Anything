from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
CONFIG.wandb = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
CONFIG.model.freeze_seg = True
CONFIG.model.multi_scale = False
CONFIG.model.imagenet_pretrain = False
CONFIG.model.imagenet_pretrain_path = "/path/to/data/model_best_resnet34_En_nomixup.pth"
CONFIG.model.batch_size = 16
# one-hot or class, choice: [3, 1]
CONFIG.model.mask_channel = 1
CONFIG.model.trimap_channel = 3

# hyper-parameter for refinement
CONFIG.model.self_refine_width1 = 30
CONFIG.model.self_refine_width2 = 15
CONFIG.model.self_mask_width = 10

# Model -> Architecture config
CONFIG.model.arch = EasyDict({})
# definition in networks/encoders/__init__.py and networks/encoders/__init__.py
CONFIG.model.arch.encoder = "res_shortcut_encoder_29"
CONFIG.model.arch.decoder = "res_shortcut_decoder_22"
CONFIG.model.arch.m2m = "sam_decoder_deep"
CONFIG.model.arch.seg = "sam"
# predefined for GAN structure
CONFIG.model.arch.discriminator = None


# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.cutmask_prob = 0
CONFIG.data.workers = 0
CONFIG.data.pha_ratio = 0.5
# data path for training and validation in training phase
CONFIG.data.train_fg = None
CONFIG.data.train_alpha = None
CONFIG.data.train_bg = None
CONFIG.data.test_merged = None
CONFIG.data.test_alpha = None
CONFIG.data.test_trimap = None
CONFIG.data.d646_fg = None
CONFIG.data.d646_pha = None
CONFIG.data.aim_fg = None
CONFIG.data.aim_pha = None
CONFIG.data.human2k_fg = None
CONFIG.data.human2k_pha = None
CONFIG.data.am2k_fg = None
CONFIG.data.am2k_pha = None
CONFIG.data.rim_pha = None
CONFIG.data.rim_img = None
CONFIG.data.coco_bg = None
CONFIG.data.bg20k_bg = None
# feed forward image size (untested)
CONFIG.data.crop_size = 1024
# composition of two foregrounds, affine transform, crop and HSV jitter
CONFIG.data.real_world_aug = False
CONFIG.data.augmentation = True
CONFIG.data.random_interp = True

### Benchmark config
CONFIG.benchmark = EasyDict({})
CONFIG.benchmark.him2k_img = '/path/to/data/HIM2K/images/natural'
CONFIG.benchmark.him2k_alpha = '/path/to/data/HIM2K/alphas/natural'
CONFIG.benchmark.him2k_comp_img = '/path/to/data/HIM2K/images/comp'
CONFIG.benchmark.him2k_comp_alpha = '/path/to/data/HIM2K/alphas/comp'
CONFIG.benchmark.rwp636_img = '/path/to/data/RealWorldPortrait-636/image'
CONFIG.benchmark.rwp636_alpha = '/path/to/data/RealWorldPortrait-636/alpha'
CONFIG.benchmark.ppm100_img = '/path/to/data/PPM-100/image'
CONFIG.benchmark.ppm100_alpha = '/path/to/data/PPM-100/matte'
CONFIG.benchmark.pm10k_img = '/path/to/data/P3M-10k/validation/P3M-500-NP/original_image'
CONFIG.benchmark.pm10k_alpha = '/path/to/data/P3M-10k/validation/P3M-500-NP/mask'
CONFIG.benchmark.am2k_img = '/path/to/data/AM2k/validation/original'
CONFIG.benchmark.am2k_alpha = '/path/to/data/AM2k/validation/mask'
CONFIG.benchmark.rw100_img = '/path/to/data/RefMatte_RW_100/image_all'
CONFIG.benchmark.rw100_alpha = '/path/to/data/RefMatte_RW_100/mask'
CONFIG.benchmark.rw100_text = '/path/to/data/RefMatte_RW_100/refmatte_rw100_label.json'
CONFIG.benchmark.rw100_index = '/path/to/data/RefMatte_RW_100/eval_index_expression.json'

# Training config
CONFIG.train = EasyDict({})
CONFIG.train.total_step = 100000
CONFIG.train.warmup_step = 5000
CONFIG.train.val_step = 1000
# basic learning rate of optimizer
CONFIG.train.G_lr = 1e-3
# beta1 and beta2 for Adam
CONFIG.train.beta1 = 0.5
CONFIG.train.beta2 = 0.999
# weight of different losses
CONFIG.train.rec_weight = 1
CONFIG.train.comp_weight = 0
CONFIG.train.lap_weight = 1
# clip large gradient
CONFIG.train.clip_grad = True
# resume the training (checkpoint file name)
CONFIG.train.resume_checkpoint = None
# reset the learning rate (this option will reset the optimizer and learning rate scheduler and ignore warmup)
CONFIG.train.reset_lr = False


# Logging config
CONFIG.log = EasyDict({})
CONFIG.log.tensorboard_path = "./logs/tensorboard"
CONFIG.log.tensorboard_step = 100
# save less images to save disk space
CONFIG.log.tensorboard_image_step = 500
CONFIG.log.logging_path = "./logs/stdout"
CONFIG.log.logging_step = 10
CONFIG.log.logging_level = "DEBUG"
CONFIG.log.checkpoint_path = "./checkpoints"
CONFIG.log.checkpoint_step = 10000


def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]


