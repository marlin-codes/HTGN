import os
import sys
import random
import time
import torch
import numpy as np
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from script.inits import prepare
from script.models.load_model import load_model
from script.config import args
from script.utils.util import logger, init_logger, mkdirs
from script.utils.data_util import loader


class Runner(object):
    def __init__(self):
        self.len = data['time_length']

    def optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def train(self):
        print('1. Initialization')
        print("2. Start training")
        t_total0 = time.time()
        t0 = time.time()
        train_edges_list = []
        train_pos_edges_list = []
        train_neg_edges_list = []
        test_pos_edges_list = []
        test_neg_edges_list = []
        logger.info(args)
        for t in range(data['time_length']):
            edge_index, pos_index, neg_index, activate_nodes, edge_weight, new_pos, new_neg = prepare(data, t)
            if t < data['time_length'] - args.testlength:
                train_edges_list.append(edge_index)
                train_pos_edges_list.append(pos_index)
                train_neg_edges_list.append(neg_index)
            else:
                test_pos_edges_list.append(pos_index)
                test_neg_edges_list.append(neg_index)

        train_edge_index = torch.cat(train_edges_list, dim=1).to(args.device)
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        best_results = [0] * 3
        if args.trainable_feat:
            self.x = None
            logger.info('using trainable feature, feature dim: {}'.format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                feature = sp.load_npz('../data/input/raw/disease/disease_lp.feats.npz').toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('using pre-defined features')
            else:
                logger.info('using one-hot feature, feature dim: {}'.format(args.nfeat))
                self.x = torch.eye(args.num_nodes).to(args.device)
            args.nfeat = self.x.size(1)

        logger.info('test length: {}'.format(args.testlength))
        self.model = load_model(args).to(args.device)
        self.model.train()
        optimizer = self.optimizer()

        min_loss = 10
        patience = 0
        min_epoch = 1
        max_patience = args.patience
        for epoch in range(1, args.max_epoch + 1):
            optimizer.zero_grad()
            z = self.model(train_edge_index, self.x)
            loss = self.model.compute_loss(z, train_edge_index)
            loss.backward()
            optimizer.step()
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_results = [epoch]
                auc_list = []
                ap_list = []
                for pos, neg, in zip(test_pos_edges_list, test_neg_edges_list):
                    auc_list.append(self.model.predict(z, pos, neg)[0])
                    ap_list.append(self.model.predict(z, pos, neg)[1])

                best_results += [np.mean(auc_list)]
                best_results += [np.mean(ap_list)]
                patience = 0
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    print('early stopping')
                    break

            if epoch % args.log_interval == 0:
                logger.info('==' * 27)
                logger.info("Epoch:{}, Loss: {:.3f}, Time: {:.2f}, patience:{}, GPU: {:.1f}MiB".format(epoch, loss,
                                                                                                       time.time() - t0,
                                                                                                       patience,
                                                                                                       gpu_mem_alloc))
                logger.info("Epoch:{:}, Best AUC: {:.4f}, AP: {:.4f}".format(best_results[0], best_results[1], best_results[2],))

        logger.info('>> Total time : %6.2f' % (time.time() - t_total0))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |test length:%d |" % (args.lr, args.nhid, args.testlength))


if __name__ == '__main__':
    random.seed(args.seed)  # random seed
    data = loader(dataset=args.dataset)  # enron10, fb, dblp
    args.num_nodes = data['num_nodes']
    init_logger(mkdirs(args.output_folder + '/log/') + args.dataset + '.txt')
    run = Runner()
    run.train()
