import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2 
import utils
import matplotlib.pyplot as plt
import shap
from torchvision.ops import sigmoid_focal_loss

from misc.losses import Focal_loss,Confusion_Loss,Supervised_Contrastive_Loss
from misc.logger_tool import *
from misc.metric_tool import *

from models.addNetworks import *


class Evaluator():
    
    def __init__(self,args,dataloaders):

        self.dataloaders = dataloaders
        self.args = args
        self.class_weights = args.class_weights
        self.best_ckpts = args.best_ckpts
        self.lr = args.lr
        self.reset_lr = args.reset_lr
        self.dataset = args.dataset
        self.data_name = args.data_name
        self.n_class = args.n_class
        self.train = args.train
        self.argloss = args.loss
        self.accumlation_steps = args.accumulation_steps
        self.alpha = args.lad_alpha
        self.walk_steps =args.walk_steps
        self.regularization = args.regularization
        self.lambda_reg = args.lambda_reg
        # define network
        self.vqvae,self.classifier = define_net(args=args)
        self.net = define_strong_net(args=args)
        self.train = args.train
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        self.disco_alpha = args.disco_alpha
        self.disco_beta = args.disco_beta
        self.disco_choice = args.disco_choice

        self.fine_tune_delta = args.fine_tune_delta    
        self.fine_tune_patience = args.fine_tune_patience
        self.fine_tuned = False
        self.patience = 0

        if self.train == 'strong_classifier':
            self.lad = True

        self.net.to(self.device)

        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)
        self.running_fairness = FairnessMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_loss = np.inf
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['val'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.pred = None
        self.pred_vis = None
        self.batch = None
        self.loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        if self.train in ['strong_classifier','classifier','standard']:
            weight_tensor = torch.tensor(self.class_weights,dtype=torch.float32,device=self.device)
            if self.argloss == 'ce': 
                self._pxl_loss = nn.CrossEntropyLoss(weight=weight_tensor)
            elif self.argloss == 'focal':
                self._pxl_loss = Focal_loss(n_class=self.n_class,alpha=self.args.focal_alpha,gamma=self.args.focal_gamma,reduction='mean')
        elif self.train == 'vqvae':
            if args.vqvae_loss == 'mse':
                self._pxl_loss = nn.MSELoss()
            elif args.vqvae_loss == 'l1':
                self._pxl_loss = nn.L1Loss()
        elif self.train == 'fairdisco':
            self._pxl_loss = [nn.CrossEntropyLoss(), Confusion_Loss(), 
            nn.CrossEntropyLoss(), Supervised_Contrastive_Loss(0.1, self.device)]
        else:
            raise NotImplemented(self.train)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            try:
                checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                        map_location=self.device,weights_only=False)
            except Exception as e:
                self.logger.write('Error occurred while loading checkpoint: %s\n' % str(e))
                return


            # update net states
            if self.train == 'vqvae':
                self.vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
            elif self.train == 'classifier':
                self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            elif self.train == 'strong_classifier' or 'standard':
                self.net.load_state_dict(checkpoint['model_strong_state_dict'])
            self.optimizer.load_state_dict(checkpoint['net_optimizer_state_dict'])
            self.exp_lr_scheduler.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            # reset lr to default
            if self.reset_lr:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.lr
                self.exp_lr_scheduler = get_scheduler(self.optimizer, self.args)

            self.net.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileExistsError(f"No checkpoint found: {ckpt_name}")
    
    def _collect_running_batch_states(self):
        
        if not self.train == 'vqvae':
            running_acc = self._update_metric()
            if self.train == 'strong_classifier' or 'standard':
                running_fairness = self._update_fairness()

            m = len(self.dataloaders['train'])
            if self.is_training is False:
                m = len(self.dataloaders['val'])

            imps, est = self._timer_update()

            ##### MESSAGE #####
            if np.mod(self.batch_id, 100) == 1:
                if self.train == 'strong_classifier' or 'standard':
                    message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f,running_EO: %.5f,' \
                    ' running_DI: %.5f, running_AP: %.5f\n' %\
                            (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                            imps*self.batch_size, est,
                            self.loss.item(), running_acc, running_fairness['EO'], running_fairness['DI'], running_fairness['AP'])
                    
                else:

                    message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                            (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                            imps*self.batch_size, est,
                            self.loss.item(), running_acc)
                self.logger.write(message)


        else:
            imps, est = self._timer_update()
            if np.mod(self.batch_id, 100) == 1:
                message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.5fh, VQvae_loss: %.5f, Vq_loss: %.5f, Perplexity: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, len(self.dataloaders['train']),
                        imps*self.batch_size, est,
                        self.loss.item(), self.vq_loss, self.perplexity)
                self.logger.write(message)

        if np.mod(self.batch_id, 300) == 1:
            vis_input = utils.make_numpy_grid(self.batch['image'][:16])
            
            if self.train == 'classifier':
                vis_perturbation = utils.make_numpy_grid(self.perturbation[:16])
                self._visualize_perturbations(vis_input,vis_perturbation)
                return

            if self.train in ['strong_classifier','classifier']:
                vis_perturbation = utils.make_numpy_grid(self.perturbation[:16])
                self._visualize_perturbations(vis_input,vis_perturbation)
                vis_pred = self._visualize_pred(self.batch['image'][:16])

            elif self.train in ['standard','fairdisco']:
                vis_pred = self._visualize_pred(self.batch['image'][:16])

            else:
                vis_pred = utils.make_numpy_grid(self.net_pred[:16])
            
            vis = np.concatenate([vis_input, vis_pred], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)

            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                            str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        if self.train in ['strong_classifier','classifier','standard','fairdisco']:
            scores = self.running_metric.get_scores()
            fairness = self.running_fairness.get_scores()
            self.epoch_acc = scores['mf1']
            self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n'\
                              'F1_benign= %.5f, F1_malignant: %.5f\n' %
                (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc, \
                    scores['F1_0'], scores['F1_1']))
            
            self.logger.write(
                f"Is_training:{self.is_training}. Epoch {self.epoch_id}/{self.max_num_epochs-1} "
                f"epoch_EO={fairness['epoch_EO']:.4f}, epoch_DI={fairness['epoch_DI']:.4f}, epoch_AP={fairness['epoch_AP']:.4f}, "
                f"avg_EO={fairness['avg_EO']:.4f}, avg_DI={fairness['avg_DI']:.4f}, avg_AP={fairness['avg_AP']:.4f}\n"
            )

            target = self.batch['label'].to(self.device).detach()
            if self.train == 'fairdisco':

                pred = self.net_pred[0].detach()
            else:
                pred = self.net_pred.detach()
            pred = torch.argmax(pred,dim=1)
            
            self.running_metric.get_cm(target.cpu().numpy(),pred.cpu().numpy())

        elif self.train == 'vqvae':
            self.logger.write('Is_training: %s. Epoch %d / %d, epoch_VQ_loss= %.5f\n' %
                (self.is_training, self.epoch_id, self.max_num_epochs-1, self.loss.item()))

    def _update_fairness(self):
        # 1. Extract data and move to device
        target = self.batch['label'].to(self.device).detach()
        fitzpatrick = self.batch['fitzpatrick'].to(self.device).detach()
        
        # 2. Process predictions
        if self.train == 'fairdisco':
            pred = self.net_pred[0].detach()
        else:
            pred = self.net_pred.detach()
        pred = torch.argmax(pred, dim=1)

        # 3. Create boolean masks
        # Protected: Fitzpatrick 4, 5, 6 | Non-protected: 1, 2, 3
        protected_mask = fitzpatrick > 3
        non_protected_mask = ~protected_mask

        # 4. Split the data

        target_prot = target[protected_mask].cpu().numpy()
        pred_prot = pred[protected_mask].cpu().numpy()

        # Non-protected group
        target_non_prot = target[non_protected_mask].cpu().numpy()
        pred_non_prot = pred[non_protected_mask].cpu().numpy()

        # 5. Update fairness tracker

        current_score = self.running_fairness.update_cm(
            pr_prot=pred_prot, 
            gt_prot=target_prot,
            pr_unprot=pred_non_prot,
            gt_unprot=target_non_prot
        )
        
        return current_score

    def _update_metric(self):
        target = self.batch['label'].to(self.device).detach()

        if self.train == 'fairdisco':
            pred = self.net_pred[0].detach()
        else:
            pred = self.net_pred.detach()

        pred = torch.argmax(pred,dim=1)

        current_score = self.running_metric.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _clear_cache(self):
        self.running_metric.clear()
        self.running_fairness.clear()

    def _forward_pass(self,batch):
        self.batch = batch

        if self.train == 'strong_classifier':
            vqvae_out = self.vqvae.encoder(batch['image'].to(self.device))
            vqvae_out = self.vqvae.pre_vq_conv(vqvae_out)
            adv_walk, _ = self.adversarial_walk(vqvae_out,steps=self.walk_steps,a=self.alpha)
            self.perturbation = self.vqvae.decoder(adv_walk)
            self.net_pred = self.net(self.perturbation)
        elif self.train == 'vqvae':
            self.vq_loss, self.net_pred, self.perplexity = self.vqvae(batch['image'].to(self.device))
        elif self.train == 'classifier':
            vqvae_out = self.vqvae.encoder(batch['image'].to(self.device))
            vqvae_out = self.vqvae.pre_vq_conv(vqvae_out)
            self.net_pred = self.net(vqvae_out)
            adv_walk, _ = self.adversarial_walk(vqvae_out,steps=self.walk_steps,a=self.alpha)
            self.perturbation = self.vqvae.decoder(adv_walk)
        elif self.train == 'standard':
            self.net_pred = self.net(batch['image'].to(self.device))
        elif self.train == 'fairdisco':
            self.net_pred = self.net(batch['image'].float().to(self.device))
            self.fitz_pred = batch['fitzpatrick']

    def eval_models(self,checkpoint_name='best_ckpt.pt'):
            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                if not self.train == 'vqvae':
                    self._forward_pass(batch)   # we need gradients to compute the grad-cam
                else:
                    with torch.no_grad():
                        self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()