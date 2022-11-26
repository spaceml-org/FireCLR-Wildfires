import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.save_models import save_config_file, accuracy, save_checkpoint
from torch import device


class SimCLR(object):
    '''
    This class was built based on Thalles Silva's implementation using Pytorch: https://github.com/sthalles/SimCLR
    '''

    def __init__(self, args_dict, optimizer, model, scheduler):
        self.args_dict = args_dict
        self.model = model.to(self.args_dict["device"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args_dict["device"])

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args_dict["batch_size"]) for i in range(self.args_dict["n_views"])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # print("labels shape and labels:")
        # print(labels.shape)
        # print(labels)
        labels = labels.to(self.args_dict["device"])

        # print("features shape: ", features.shape)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # print("Similarity matrix shape: ", similarity_matrix.shape)

        # assert similarity_matrix.shape == (
        #     self.args_dict.n_views * self.args_dict.batch_size, self.args_dict.n_views * self.args_dict.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args_dict["device"])
        # print("Mask: ", mask)
        labels = labels[~mask].view(labels.shape[0], -1)

        # print("Masked labels:")

        # print(labels, labels.shape)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # print("Masked similarity matrix and its shape: ", similarity_matrix, similarity_matrix.shape)
        assert similarity_matrix.shape == labels.shape
        
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # print("Masked positives similarity matrix and its shape: ", positives, positives.shape)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        # print("Masked negatives similarity matrix and its shape: ", negatives, negatives.shape)

        logits = torch.cat([positives, negatives], dim=1)
        # print("logits shape: ", logits.shape)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args_dict["device"])
        # print("labels and its shape: ", labels, labels.shape)

        logits = logits / self.args_dict["temperature"]
        # print("Output logits and input temperature: ", self.args_dict["temperature"],logits, logits.shape)

        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args_dict["fp16_precision"])

        # save config file
        save_config_file(self.writer.log_dir, self.args_dict)

        n_iter = 0
        logging.info("Start SimCLR training for %d epochs." %self.args_dict["epochs"])
        #logging.info(f"Training with gpu: {self.args_dict.disable_cuda}.")

        for epoch_counter in range(self.args_dict["epochs"]):
            for image_1, image_2 in tqdm(train_loader):
                images = torch.cat((image_1,image_2), dim=0)

                images = images.to(self.args_dict["device"],dtype=torch.float)

                with autocast(enabled=self.args_dict["fp16_precision"]):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args_dict["log_every_n_steps"] == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter) #update get_last_lr()?
                
                if n_iter % self.args_dict["checkpoint_every_n_steps"] == 0:
                    # save model checkpoints
                    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(n_iter//self.args_dict["checkpoint_every_n_steps"])
                    save_checkpoint({
                        'step': n_iter,
                        'arch': self.args_dict["arch"],
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                    logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
                    
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args_dict["epochs"])
        save_checkpoint({
            'epoch': self.args_dict["epochs"],
            'arch': self.args_dict["arch"],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")