from args import args
from logger import log_step, log, log_confusion_matrix, log_scalar, log_text
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from data import get_loaders
from model import Unet, ResNet
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import datetime
import os
import random
import time

class Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Loss, self).__init__()
        self.margin = margin
        self.cls_loss = nn.CrossEntropyLoss()
        
    def classification_loss(self, output, labels):
        output = output[:, args.num_supports:]
        labels = labels.to(output.device)
        preds = torch.argmax(output, dim=-1)
        accuracy = torch.sum(torch.eq(preds, labels[:, args.num_supports:].long())).float() / preds.numel()
        cls_loss = self.cls_loss(output, labels[:, args.num_supports:].long()).mean()
        return cls_loss, accuracy
    
    def contrastive_loss(self, embeddings, labels):
        batch_size, num_samples, feature_dim = embeddings.size()
        embeddings = embeddings.view(-1, feature_dim)  # Flatten to [batch_size * num_samples, feature_dim]
        labels = labels.view(-1) 
        
        # Compute pairwise distances
        euclidean_distance = F.pdist(embeddings, p=2)
        
        # Create contrastive labels: 1 for similar pairs, 0 for dissimilar pairs
        labels = labels.unsqueeze(1)
        contrastive_labels = labels.eq(labels.t()).float()
        # Get indices for the upper triangular matrix without diagonal
        triu_indices = torch.triu_indices(batch_size * num_samples, batch_size * num_samples, offset=1)
        contrastive_labels = contrastive_labels[triu_indices[0], triu_indices[1]]
        
        # Calculate losses for similar and dissimilar pairs
        positive_pairs = contrastive_labels * euclidean_distance
        negative_pairs = (1 - contrastive_labels) * torch.clamp(self.margin - euclidean_distance, min=0.0)
        
        # Calculate final loss
        loss = torch.mean(positive_pairs + negative_pairs)
        return loss
    
    
    def forward(self, output, embeddings, labels):
        cls_loss, accuracy = self.classification_loss(output, labels)
        contrastive_loss = self.contrastive_loss(embeddings, labels)
        return cls_loss, contrastive_loss, accuracy

        
class ModelTrainer:
    def __init__(self, enc_module, unet_module, data_loader):
        self.data_loader = data_loader
        
        self.enc_module = enc_module.to(args.device)
        self.unet_module = unet_module.to(args.device)
        
        self.module_params = list(self.enc_module.parameters()) + list(self.unet_module.parameters())
        
        self.optimizer = optim.Adam(params=self.module_params, lr=args.lr, weight_decay=args.weight_decay)
        self.loss = Loss()
        
        self.best_checkpoint = os.path.join(args.log_dir, 'best.pth.tar')
        self.last_checkpoint = os.path.join(args.log_dir, 'last.pth.tar')
        
        self.global_step = 0
        self.best_acc = 0
        self.best_step = 0

        if args.resume:
            self.load_checkpoint()

    def train(self):
        start_time = time.time()
        for iter in range(self.global_step + 1, args.train_iteration + 1):
            self.optimizer.zero_grad()
            self.global_step = iter

            task_batch = self.load_data('train', iter)
            
            data, labels, edges = self.process_task_data(task_batch)
            
            output, embeddings = self.unet_module(edges, data)
           
            cls_loss, contrastive_loss, accuracy = self.loss(output, embeddings, labels)
            
            cls_loss.backward(retain_graph=True) # retain_graph=True => to keep the computation graph for further backward passes
            contrastive_loss.backward()
            self.optimizer.step()

            if self.global_step % args.log_step == 0:
                self.log_stats(start_time, accuracy, cls_loss, contrastive_loss)
                start_time = time.time()
                

            if self.global_step % args.test_interval == 0:
                self.eval('val', True)
                if self.best_acc == 1:
                    log('[INFO]  The training stopped because the validation accuracy acheived 100%.')
                    break
                if (self.global_step - self.best_step) > args.early_stop_gap:
                    log(f'[INFO]  The training stopped because the validation accuracy has not been improved during the last {args.early_stop_gap} epochs.')
                    break
   
    def eval(self, partition='val', log_flag=True):
        self.enc_module.eval()
        self.unet_module.eval()
        
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        all_preds = []
        all_labels = []

        for iter in range(args.test_iteration // args.batch_size):
            task_batch = self.load_data(partition, iter)
            data, labels, edges = self.process_task_data(task_batch)
            
            output, _ = self.unet_module(edges, data)
            
            output = output[:, args.num_supports:]
            labels = labels.to(output.device)[:, args.num_supports:]
            
            preds = torch.argmax(output, dim=-1)
            accuracy = torch.sum(torch.eq(preds, labels.long())).float() / preds.numel()
            
            labels = labels.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()
            
            accuracies.append(accuracy.item())
            precisions.append(precision_score(labels, preds, average='macro', zero_division=0))
            recalls.append(recall_score(labels, preds, average='macro', zero_division=0))
            f1s.append(f1_score(labels, preds, average='macro', zero_division=0))

            all_preds.extend(preds)
            all_labels.extend(labels)

        cm = confusion_matrix(all_labels, all_preds)
        acc_mean = np.mean(accuracies)
        acc_std = np.std(accuracies)
        precision_mean = np.mean(precisions)
        recall_mean = np.mean(recalls)
        f1_mean = np.mean(f1s)

        is_best = acc_mean > self.best_acc
        if is_best:
            self.best_acc = acc_mean
            self.best_step = self.global_step
            
        self.save_checkpoint(is_best)

        if log_flag:
            log('------------------------------')
            log('eval: count: %d, best: %.2f%% (%d epoch), val accuracy: %.2f%%, std: %.2f%%, precision: %.2f%%, recall: %.2f%%, f1: %.2f%%' %
                (iter, self.best_acc * 100, self.best_step, acc_mean * 100, acc_std * 100, precision_mean * 100, recall_mean * 100, f1_mean * 100))
            log_confusion_matrix(cm)
            log('------------------------------')

        return [acc_mean, precision_mean, recall_mean, f1_mean, cm]

    def test(self):
        if os.path.isfile(self.best_checkpoint):
            self.load_checkpoint(best=True)
            self.enc_module.eval()
            self.unet_module.eval()

            metrics = self.eval('test', False)
            accuracy, precision, recall, f1, cm = metrics

            eval_str ='Best Model Testing: {}\nAccuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1 Score: {:.2f}%'.format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), accuracy * 100, precision * 100, recall * 100, f1 * 100)
            log('******************************')
            log(eval_str)
            log_confusion_matrix(cm)
            log('******************************')
            with open(os.path.join(args.log_dir, 'best_eval.txt'), 'a') as f:
                f.write(eval_str + "\n")
        else:
            log("No checkpoint found at '{}'".format(self.best_checkpoint))

    def load_data(self, partition, iteration=None):
        return self.data_loader[partition].get_batch(
            num_tasks=args.batch_size,
            num_ways=args.num_ways,
            num_shots=args.num_shots,
            num_queries=args.num_queries // args.num_ways,
            seed=iteration + args.seed
        )

    def process_task_data(self, task_batch):
        support_data, support_label, query_data, query_label = task_batch
        num_supports = args.num_ways * args.num_shots
        
        data = torch.cat([support_data, query_data], dim=1)
        label = torch.cat([support_label, query_label], dim=1)
        
        num_samples = label.size(1)
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)
        
        edges = torch.eq(label_i, label_j).float().to(args.device)
        edges[:, num_supports:, :] = 0.5
        edges[:, :, num_supports:] = 0.5
        for i in range(args.num_queries):
            edges[:, num_supports + i, num_supports + i] = 1.0
            
        num_queries = data.size(1) - num_supports

        encoded_data = [self.enc_module(data.squeeze(1)) for data in data.chunk(data.size(1), dim=1)]
        encoded_data = torch.stack(encoded_data, dim=1)  # [batch_size, num_samples, feat_dim]

        # Generating one-hot labels
        one_hot_label = torch.eye(args.num_ways, device=args.device)[label.long()].to(args.device)
        one_hot_label = one_hot_label[:, :num_supports + num_queries, :]

        data = torch.cat([encoded_data, one_hot_label], dim=-1)
        
        return data, label, edges   
    
    def log_stats(self, start_time, accuracy, cls_loss, cont_loss):
        total_time = int(time.time() - start_time)
        log_scalar('accuracy', accuracy, self.global_step)
        log_scalar('cls_loss', cls_loss, self.global_step)
        log_scalar('cont_loss', cont_loss, self.global_step)
        log_text('time', f'{total_time}s', self.global_step)

        log_step(self.global_step, args.train_iteration)
        
    def save_checkpoint(self, best=False):
        state = {
            'iteration': self.global_step,
            'enc_module_state_dict': self.enc_module.state_dict(),
            'unet_module_state_dict': self.unet_module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_step': self.best_step,
        }
        torch.save(state, self.last_checkpoint)
        if best:
            torch.save(state, self.best_checkpoint)

    def load_checkpoint(self, best=False):
        checkpoint_path = self.best_checkpoint if best else self.last_checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.global_step = checkpoint['iteration']
            self.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
            self.unet_module.load_state_dict(checkpoint['unet_module_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_acc = checkpoint['best_acc']
            self.best_step = checkpoint['best_step']
            log("[INFO] Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, self.global_step))
        else:
            log("[INFO] No checkpoint found to resume")

    

def set_exp_name():
    return 'D-{}_N-{}_K-{}'.format(args.dataset, args.num_ways, args.num_shots)
     

if __name__ == '__main__':
    # data and fsl parameters
    args.device = 'cuda:0' if args.device is None else args.device
    args.dataset_root = 'data'
    args.dataset = args.dataset
    args.num_ways = 5 if args.num_ways is None else args.num_ways
    args.num_shots = 5 if args.num_shots is None else args.num_shots
    args.num_queries = args.num_ways
    args.num_supports = args.num_ways*args.num_shots
    args.num_gpus = 1 if args.num_gpus is None else args.num_gpus
    
    # train, test parameters
    args.train = False if args.train is None else True
    args.test = False if args.test is None else True
    args.train_iteration = 100000 if args.train_iteration is None else args.train_iteration
    args.test_iteration = 1000 if args.test_iteration is None else args.test_iteration
    args.batch_size = 8 if args.batch_size is None else args.batch_size
    args.test_interval = 500 if args.test_interval is None else args.test_interval
    args.early_stop_gap = 10000 if args.early_stop_gap is None else args.early_stop_gap
    
    args.log_step = 10 if args.log_step is None else args.log_step
    args.resume = False if args.resume is None else True

    args.lr = 1e-5 if args.lr is None else args.lr
    args.weight_decay = 1e-6 if args.weight_decay is None else args.weight_decay

    args.experiment = set_exp_name() if args.experiment is None else args.experiment

    # model parameter related
    args.emb_size = 32
    args.in_dim = args.emb_size + args.num_ways
    args.pooling_ratios = [0.6, 0.5] # Higher ratios for less aggressive pooling
    args.dropout = 0.1 if args.dropout is None else args.dropout

    
    # seeding
    args.seed = 1998 if args.seed is None else args.seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

    # logging dir
    args.log_dir = os.path.join('logs', args.experiment)
    args.log_file = os.path.join(args.log_dir, 'log.txt')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)



    unet_module = Unet(args.pooling_ratios, args.in_dim,  args.num_ways,  args.num_queries, args.dropout)
    enc_module = ResNet(args.emb_size)
    
    data_loader = get_loaders(os.path.join(args.dataset_root, args.dataset), args.device)
    trainer = ModelTrainer(enc_module=enc_module, unet_module=unet_module, data_loader=data_loader)
    

    log('Configurations')
    log('-' * 30)
    for k, v in args.__dict__.items():
        log('%s=%s' % (k, v))
    log('-' * 30)
    
    if args.train:
        trainer.train()
    if args.test:
        trainer.test()





'''
python3 train.py --dataset WHU-RS19-5-4-1 --num_shots 5 --batch_size 32 --dropout 0.1 --train 1 --test 1

python3 train.py --dataset WHU-RS19-8-1-1 --num_shots 1 --batch_size 32 --dropout 0.1 --train 1 --test 1

'''