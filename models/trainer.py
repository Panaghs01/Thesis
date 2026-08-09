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


class Trainer():

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

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        if args.train in ['strong_classifier','standard']:
            self._freeze_model()
        elif args.train == 'vqvae':
            self.net = self.vqvae
        elif args.train == 'classifier':
            self.net = self.classifier.to(self.device)
            self.vqvae.to(self.device)

        # define optimizers
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(),
                                          lr=self.lr, momentum=0.9,
                                            weight_decay=5e-4)
        elif args.optimizer == 'adam':
            if self.train in ['standard','strong_classifier']:
                self.optimizer = optim.AdamW(self.net.parameters(),lr=self.lr,weight_decay=1e-3)
            elif self.train in ['fairdisco']:
                self.optimizer = optim.AdamW(self.net.disco.parameters(),lr=self.lr,weight_decay=1e-3)
            else:
                self.optimizer = optim.AdamW(self.net.parameters(),
                                           lr=self.lr,weight_decay=1e-3)



        # define lr schedulers
        self.exp_lr_scheduler = get_scheduler(self.optimizer, args)

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
        self.steps_per_epoch = len(dataloaders['train'])
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
                self._pxl_loss = Focal_loss(n_class=self.n_class,alpha=0.75,gamma=2,reduction='mean')
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

    def _update_lr_schedulers(self):
        if isinstance(self.exp_lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.exp_lr_scheduler.step(self.epoch_acc)
        else:
            self.exp_lr_scheduler.step()

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
            print('training from scratch...')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name=f"{self.train}_last_ckpt.pt")

        if self.train == 'vqvae':
            message = 'Latest model updated. Epoch loss=%.4f, Best loss:=%.4f (at epoch %d)\n' \
                % (self.loss,self.best_loss,self.best_epoch_id)
        else:
            message = 'Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n' \
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id)
        self.logger.write(message)
        self.logger.write('\n')

        # update the [best model] (based on eval acc)
        if self.train == 'vqvae':
            if self.loss < self.best_loss:
                self.best_loss = self.loss
                self.best_epoch_id = self.epoch_id
                self.best_val_acc = self.epoch_acc
                self._save_checkpoint(ckpt_name=f"best_ckpt_{self.train}.pt")
        else:
            if self.epoch_acc > self.best_val_acc:
                self.best_epoch_id = self.epoch_id
                self.best_val_acc = self.epoch_acc
                self._save_checkpoint(ckpt_name=f"best_ckpt_{self.train}.pt")
                self.logger.write("*"*10+'Best model updated!\n')
            self.logger.write('\n')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self,imgs):
        imgs = imgs.to(self.device)
        if self.train == 'strong_classifier' or 'standard':
            gradients = self.net.get_activations_gradient()
            pooled_gradients = torch.mean(gradients,dim=[0,2,3])
            activations = self.net.get_activations(imgs).detach()

            channels = activations.shape[1]
            for i in range(channels):
                activations[:,i,:,:] *= pooled_gradients[i]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = torch.relu(heatmap)

            max_val = torch.max(heatmap)
            if max_val.item() != 0:
                heatmap /= max_val

            heatmap = heatmap.unsqueeze(1)
            resized_heatmap = torch.nn.functional.interpolate(heatmap, size=imgs.shape[2:], mode='bilinear', align_corners=False)

            resized_heatmap = np.uint8(resized_heatmap.cpu().numpy() * 255)
            grid_heatmap = utils.make_numpy_grid(torch.from_numpy(resized_heatmap))
            grid_imgs = utils.make_numpy_grid(imgs.cpu())
            grid_heatmap = cv2.applyColorMap(grid_heatmap, cv2.COLORMAP_JET)

            pred_vis = grid_heatmap * 0.4 + grid_imgs 

        else:
            pred_vis = self.net_pred * 255

        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        if 'best_ckpt' in ckpt_name:
            torch.save({
                'epoch_id': self.epoch_id,
                'best_val_acc': self.best_val_acc,
                'best_epoch_id': self.best_epoch_id,
                'model_strong_state_dict': self.net.state_dict(),
                'net_optimizer_state_dict': self.optimizer.state_dict(),
                'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler.state_dict(),
                'vqvae_state_dict': self.net.state_dict() if self.train == 'vqvae' else None,
                'classifier_state_dict': self.net.state_dict() if self.train == 'classifier' else None,
        }, os.path.join(self.best_ckpts, ckpt_name))

        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_strong_state_dict': self.net.state_dict(),
            'net_optimizer_state_dict': self.optimizer.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler.state_dict(),
            'vqvae_state_dict': self.net.state_dict() if self.train == 'vqvae' else None,
            'classifier_state_dict': self.net.state_dict() if self.train == 'classifier' else None,
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_metric(self):
        target = self.batch['label'].to(self.device).detach()

        if self.train == 'fairdisco':
            pred = self.net_pred[0].detach()
        else:
            pred = self.net_pred.detach()

        pred = torch.argmax(pred,dim=1)

        current_score = self.running_metric.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score
    
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

    def _visualize_perturbations(self,vis_input,vis_perturbation):
        if not self.train == 'vqvae':
            
            vis = np.concatenate([vis_input, vis_perturbation], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)

            file_name = os.path.join(
                self.vis_dir, 'perturbation_'+str(self.is_training)+'_'+
                                str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        if self.train in ['strong_classifier','classifier','standard']:
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

    def adversarial_walk(self,vqvae_out,steps=10,a=0.2):
        h_delta = vqvae_out.clone().detach().requires_grad_(True)
        e = 1e-4
        
        for _ in range(steps):
            prediction = self.classifier(h_delta)
            prediction = torch.softmax(prediction,dim=1)
            entropy = -torch.special.entr(prediction+e).sum(dim=1).mean()

            grad = torch.autograd.grad(entropy, h_delta, create_graph=False)[0]
            #delta = (grad - grad.mean()) / (grad.std() + e)

            delta = torch.sign(grad)   # really small grad due to std. Try sign
            h_delta = (h_delta + a*delta).detach().requires_grad_(True)

        _,h_delta,perplexity,_ = self.vqvae.vq(h_delta)

        return h_delta, perplexity

    def _update_training_acc_curve(self):
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, self.epoch_acc)
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        self.VAL_ACC = np.append(self.VAL_ACC, self.epoch_acc)
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

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

    def _backward(self):

        if self.train == 'vqvae':
            gt = self.batch['image'].to(self.device).float()
            self.loss = self._pxl_loss(self.net_pred.float(), gt)  + self.vq_loss
        elif self.train in ['strong_classifier','classifier','standard']:
            gt = self.batch['label'].to(self.device).long()
            self.loss = self._pxl_loss(self.net_pred.float(), gt)
        elif self.train == 'fairdisco':
            label_c, label_t = self.batch['label'].to(self.device),self.batch['fitzpatrick'].to(self.device)
            label_t -= label_t-1
            loss0 = self._pxl_loss[0](self.net_pred[0],label_c)
            loss1 = self._pxl_loss[1](self.net_pred[1],label_t)
            loss2 = self._pxl_loss[2](self.net_pred[2],label_t)
            loss3 = self._pxl_loss[3](self.net_pred[3],label_c)

            self.loss = loss0 + loss1*self.disco_alpha + loss2 + loss3*self.disco_beta
        
        self.loss.backward()
    
    def train_models(self):
        self._load_checkpoint(ckpt_name=f"{self.train}_last_ckpt.pt")

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net.train()  # Set model to training mode
            self.net.to(self.device)
            if self.train == 'classifier':
                self.vqvae.eval()
                self.vqvae.to(self.device)
            if self.train in ['strong_classifier','standard']:
                self.vqvae.eval()
                self.vqvae.to(self.device)
                self.classifier.eval()
                self.classifier.to(self.device)
            # Iterate over data.
            self.logger.write('lr: %0.8f\n' % self.optimizer.param_groups[0]['lr'])

            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):


                self._forward_pass(batch)               
                # update G
                self._backward()

                #print('gradient sum: ',sum(p.grad.norm().item() for p in self.net.parameters()))
                if self.accumlation_steps > 0:
                    if (self.batch_id + 1) % self.accumlation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self._collect_running_batch_states()
                self._timer_update()

                del batch
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            

            torch.cuda.empty_cache()

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

            if self.train in ['strong_classifier', 'standard']:
                self._check_patience()
            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()

    def resize(arr,shape):
        arr = arr.astype(np.uint8)
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        #print(f"array size {arr[:,:,0,:].shape}, target shape {shape}")
        return cv2.resize(arr, (shape[1], shape[0]))
    
    def _freeze_model(self):
        for param in self.net.features_conv.parameters():
            param.requires_grad = False

    def _fine_tune_model(self):
        # Unfreeze all feature layers for fine-tuning
        for param in self.net.features_conv.parameters():
            param.requires_grad = True

    def _check_patience(self):

        # Do nothing if we have already fine‑tuned once.
        if self.fine_tuned:
            return

        # Evaluate stagnation.
        if (self.epoch_acc < self.best_val_acc + self.fine_tune_delta) \
            and (self.patience < self.fine_tune_patience):
            self.patience += 1
            self.logger.write("\nIMPATIENT")
        else:
            self.logger.write("\nPATIENT")
            # A successful epoch – reset patience.
            self.patience = 0

        # Trigger fine‑tune once the patience ceiling is hit.
        if self.patience == self.fine_tune_patience:
            # Clear the 999 sentinel used in the old implementation.
            self.logger.write("\n\n\n" + 10*'*' + " FINETUNING!! ")
            self._fine_tune_model()      # unfreeze all feature layers
            self.fine_tuned = True       # prevent re‑trigger
            self.patience = 0            # reset counter
    
    def _get_background(self,loader,samples=100,lad=False):
        imgs = []

        for batch in loader:

            if lad:
                x = batch['image'].to(self.device)

                z = self.vqvae.encoder(x)
                z = self.vqvae.pre_vq_conv(z)

                adv_walk, _ = self.adversarial_walk(
                    z,
                    steps=self.walk_steps,
                    a=self.alpha
                )

                perturb = self.vqvae.decoder(adv_walk)

                imgs.append(perturb)
            else:
                imgs.append(batch['image'])

            if sum(x.shape[0] for x in imgs) >= samples:
                break
        
        return torch.cat(imgs,dim=0)[:samples]

    def _get_shap_values(self):

        background = self._get_background(self.dataloaders['train'],samples=100,lad=self.lad)
        tests = self._get_background(self.dataloaders['val'],samples=4,lad=self.lad)
        explainer = shap.GradientExplainer(self.net,background)

        shap_values = explainer.shap_values(tests)

        shap.image_plot(shap_values,tests)

