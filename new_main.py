import os
from argparse import ArgumentParser
from thop import profile



import utils
from models.trainer import Trainer



def train(args):
    """ Train the model. """
    dataloaders = utils.get_dataloaders(args)
    model = Trainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    pass


if __name__ == '__main__':


    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='-1',
                help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='Fitzpatrick17k', type=str)
    parser.add_argument('--data_name', default='Fitzpatrick17k', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--class_weight',default=0.2,type=float,help='Subtract from 1 to get class 1 weight.')

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--train', default="vqvae", type=str, help='strong_classifier |' \
                                'vqvae (only train vqvae) | classifier (only train classifier)')
    parser.add_argument('--strong_classifier', default='base_resnet18', type=str,
                        help='base_resnet18 | base_efficientnet_b0 | base_densenet121')
    parser.add_argument('--loss', default='mse', type=str)

    #vqvae
    parser.add_argument('--vqvae_hiddens', default=128, type=int, help='hidden dimension for vqvae')
    parser.add_argument('--vqvae_residual_hiddens', default=128, type=int, help='hidden dimension for vqvae residual stack')
    parser.add_argument('--vqvae_residual_layers', default=2, type=int, help='number of layers for vqvae residual stack')
    parser.add_argument('--vqvae_num_embeddings', default=1024, type=int, help='number of embeddings for vqvae')
    parser.add_argument('--vqvae_embedding_dim', default=64, type=int, help='embedding dimension for vqvae')
    parser.add_argument('--vqvae_commitment_cost', default=0.25, type=float, help='commitment cost for vqvae')
    parser.add_argument('--vqvae_loss', default='mse', type=str, help='loss for vqvae, mse or l1')

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--reset_lr', default=0, type=int, help='reset learning rate to default value before training')
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--accumulation_steps', default=0, type=int)
    parser.add_argument('--vis_dir', default='vis', type=str)

    # LAD params
    parser.add_argument('--lad_alpha',default=0.2,type=float,help='Perturbation intensity')
    parser.add_argument('--walk_steps',default=10,type=int,help='Adversarial walk step count')
    parser.add_argument('--fine_tune_patience',default=5,type=int,help='After these epochs model will train all layers')
    parser.add_argument('--fine_tune_delta',default=0.05,type=float)

    args = parser.parse_args()
    
    print(args.gpu_ids)

    if args.reset_lr:
        default_lr = parser.get_default('lr')
        args.lr = default_lr

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)

    # best checkpoints for models
    args.best_ckpts = os.path.join(args.checkpoint_root,'best_checkpoints')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_ckpts, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name, args.train)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    #test(args)
