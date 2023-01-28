import os
import os.path as osp
import sys
import numpy as np

class Config:
    ## dataset
    trainset_3d = []
    trainset_2d = [] 
    testset = 'Rhp' #Rhp, STB
    ## model setting
    resnet_type = 50 
    
    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64)
    sigma = 2.5
    train_batch_size = 32 

    ## testing config
    test_batch_size = 20
    scale = 1.0
    rot = 0.0
    
    ## others
    num_thread = 12
    gpu_ids = '1'
    num_gpus = 1
    stage = 'lixel' # lixel, param
    continue_train = False
    rnn = False

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__)) #'/data/sglee/handAR' #osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    baseline = False
    onlyImgFeat = False
    frontClipFeat = False
    MultiplyFeat=False
    AddFeat=False
    AvgSum=False
    Contrastive=True
    Cont_add=False 
    Ablation_add = False
    Ablation_normal_aug = False
    Ablation_normal_aug_clip = False

    adain1 = False
    adain2 = False
    adain3 = False

    img_weight = 0.8 
    above_folder_name = "weight" 

    data_dir = osp.join(root_dir, '../dataset/')
    output_dir = osp.join(root_dir, f'{above_folder_name}')
    model_dir = osp.join(root_dir, f'{above_folder_name}') 
    load_model_dir = osp.join(root_dir,  f'{above_folder_name}') #

    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = "/data/sglee/colorhandpose3d-pytorch/main/manopth" 
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    
    def set_args(self, gpu_ids, stage='param', continue_train=False, rnn=False, tex=False, finetune=True):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        # extend training schedule
        if self.stage == 'param':
            self.lr_dec_epoch = [x+5 for x in self.lr_dec_epoch]
            self.end_epoch = self.end_epoch + 5
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

        self.rnn = rnn
        self.tex = tex
        self.finetune = finetune

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from dir import add_pypath, make_folder
add_pypath(osp.join(cfg.root_dir))
add_pypath(osp.join(cfg.data_dir))
add_pypath('../..')
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
make_folder(cfg.load_model_dir)
make_folder(cfg.model_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)