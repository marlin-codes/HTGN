import os
import sys
import random
import time
import math
import torch
import numpy as np
from math import isnan
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from script.inits import prepare
from script.models.load_model import load_model
from script.loss import ReconLoss, VGAEloss
from script.config import args
from script.utils.util import init_logger, logger
from script.utils.data_util import loader,prepare_dir

log_interval = 50


class Runner(object):
    def __init__(self):
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        if args.trainable_feat:
            self.x = None
            logger.info("using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                feature = sp.load_npz('../data/input/raw/disease/disease_lp.feats.npz').toarray()
                self.x = [torch.from_numpy(feature).float().to(args.device)] * len(self.train_shots)
                logger.info('using pre-defined feature')
            else:
                self.x = [torch.eye(args.num_nodes).to(args.device)] * len(self.train_shots)
                logger.info('using one-hot feature')
            args.nfeat = self.x[0].size(1)

        self.model = load_model(args).to(args.device)
        self.loss = ReconLoss(args)
        logger.info('tota length: {}, test length: {}'.format(self.len, args.testlength))

    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def train(self):
        print('1. Initialization')
        minloss = 10
        min_epoch = 1
        max_patience = args.patience
        patience = 0
        optimizer = self.optimizer()
        print("2. Start training")
        t_total0 = time.time()
        best_results = [0] * 5
        for epoch in range(1, args.max_epoch + 1):
            self.model.train()
            t0 = time.time()
            epoch_losses = []
            self.model.init_hiddens()
            optimizer.zero_grad()
            embeddings = self.model([data['edge_index_list'][ix].long().to(args.device) for ix in self.train_shots], self.x)
            self.model.update_hiddens_all_with(embeddings[-1])
            # compute loss
            for t, z in enumerate(embeddings):
                edge_index = prepare(data, t)[0]
                epoch_loss = self.loss(z, edge_index)
                epoch_losses.append(epoch_loss)
            sum(epoch_losses).backward()
            optimizer.step()
            # update the best results.
            average_epoch_loss = np.mean([epoch_loss.item() for epoch_loss in epoch_losses])
            if average_epoch_loss < minloss:
                minloss = average_epoch_loss
                best_results = self.test(epoch, embeddings[-1])
                patience = 0
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    print('early stopping')
                    break
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            if epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("Epoch:{}, Loss: {:.3f}, Time: {:.2f}, GPU: {:.1f}MiB".format(epoch, average_epoch_loss,
                                                                                          time.time() - t0,
                                                                                          gpu_mem_alloc))
                logger.info(
                    "Epoch:{:}, Best AUC: {:.4f}, AP: {:.4f}, New AUC: {:.4f}, New AP: {:.4f}".format(best_results[0],
                                                                                                      best_results[1],
                                                                                                      best_results[2],
                                                                                                      best_results[3],
                                                                                                      best_results[4]))
            if isnan(epoch_loss):
                break
        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (
            args.lr, args.nhid, args.nb_window))

    def test(self, epoch, embeddings=None):
        auc_list, ap_list = [], []
        auc_new_list, ap_new_list = [], []
        if embeddings is not None:
            embeddings = embeddings.detach()
        else:
            print('embedding is not exist')
        self.model.eval()
        self.model.to(args.device)
        for t in self.test_shots:
            edge_index, pos_edge, neg_edge = prepare(data, t)[:3]
            new_pos_edge, new_neg_edge = prepare(data, t)[-2:]
            auc, ap = self.loss.predict(embeddings, pos_edge, neg_edge)
            auc_new, ap_new = self.loss.predict(embeddings, new_pos_edge, new_neg_edge)
            auc_list.append(auc)
            ap_list.append(ap)
            auc_new_list.append(auc_new)
            ap_new_list.append(ap_new)
        if epoch % args.log_interval == 0:
            logger.info(
                'Epoch:{}, average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_list), np.mean(ap_list)))
            logger.info('Epoch:{}, average AUC: {:.4f}; average AP: {:.4f}'.format(epoch, np.mean(auc_new_list),
                                                                                   np.mean(ap_new_list)))
        return epoch, np.mean(auc_list), np.mean(ap_list), np.mean(auc_new_list), np.mean(ap_new_list)


if __name__ == '__main__':
    random.seed(args.seed)  # random seed
    data = loader(dataset=args.dataset)  # enron10, fb, dblp
    args.num_nodes = data['num_nodes']
    log_folder = prepare_dir(args.output_folder)  # 2.create folder
    init_logger(log_folder + args.dataset + '.txt')
    run = Runner()
    run.train()
