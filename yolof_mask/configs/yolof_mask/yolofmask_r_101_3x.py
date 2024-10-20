from yolof_mask.configs.yolof_mask.train import train
from yolof_mask.configs.yolof.optim import YOLOF_SGD as optimizer
from yolof_mask.configs.yolof.coco_schedule import lr_multiplier_3x_b10 as lr_multiplier
from yolof_mask.configs.yolof.coco_dataloader import dataloader
from yolof_mask.configs.yolof_mask.yolofmask_r_101 import model

default_batch_size = 16
batch_size = 10

dataloader.train.mapper.use_instance_mask = True
dataloader.train.total_batch_size = batch_size
dataloader.evaluator.dataset_name = 'coco2017_val'
dataloader.train.dataset.names = ('coco2017_train', 'coco2017_val')
dataloader.test.dataset.names = 'coco2017_val'

train['output_dir'] = "./output_yolofmask_r_101_3x"
train['max_iter'] = 90000 * 3 * default_batch_size // batch_size
train['eval_period'] = 5000 * 3 * default_batch_size // batch_size
train['best_checkpointer']['val_metric'] = "segm/AP50"
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
train['device'] = 'cuda:2'

model.num_classes = 80
model.yolof.num_classes = 80
model.mask_head.num_classes = 80
model.yolof.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01