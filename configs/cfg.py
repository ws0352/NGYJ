import yaml
import torch


class Config:
    def __init__(self, current_path, name):
        super().__init__()

        cfg = None
        with open(current_path + "\\configs\\" + name, 'r') as f:
            cfg = yaml.safe_load(f.read())
        self.dataset_path = current_path + cfg['dataset']['path']
        self.batch_size = cfg['dataset']['batch_size']
        self.num_workers = cfg['dataset']['num_workers']
        self.num_classes = cfg['dataset']['num_classes']
        self.folder = cfg['train']['folder']
        self.n_epochs = cfg['train']['n_epochs']
        self.lr = cfg['train']['lr']
        self.verbose = cfg['train']['verbose']
        self.verbose_step = cfg['train']['verbose_step']
        self.step_scheduler = cfg['train']['step_scheduler']
        self.validation_scheduler = cfg['train']['validation_scheduler']
        self.SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.scheduler_params = dict(
            mode='min',
            factor=0.5,
            patience=1,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08
        )
