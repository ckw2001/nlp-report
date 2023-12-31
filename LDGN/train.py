import torch
import torch.utils.data

import os
import argparse
import pickle
import time
import random
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import precision_score, f1_score
from model.CL_loss import Contrastive_loss
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KDTree
import codecs
import json
from model.ldgn import Label_Attention, GCN_v1
import torch
import torch.nn as nn
from model.ldgn import BiLSTM
from model.ldgn import LDGN
import preprocess
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='train.py')
# opts.model_opts(parser)
parser.add_argument('--batch_size', type=int, default=32, required=False)
# parser.add_argument('--valid_batch_size', type=int, default=1, required=False)
parser.add_argument('--max_seq_length', type=int, default=256, required=False)
parser.add_argument('--seed', type=int, default=42, required=False)
parser.add_argument('--epochs', type=int, default=10, required=False)

opt = parser.parse_args()

# config = utils.read_config(opt.config)
# torch.manual_seed(opt.seed)
# random.seed(opt.seed)
# np.random.seed(opt.seed)
# opts.convert_to_config(opt, config)

# cuda
use_cuda = torch.cuda.is_available()
opt.use_cuda = use_cuda
if use_cuda:
    device = "cuda:0"
    torch.cuda.set_device("cuda:0")
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

# with codecs.open(opt.label_dict_file, 'r', 'utf-8') as f:
#     label_dict = json.load(f)

vocab_npa, embs_npa = preprocess.load_embedding()
opt.vocab_mapping = vocab_npa
opt.embedding = embs_npa


def load_data():
    print('loading data...\n')
    train_data_path = '../datasets/AAPD/train.tsv'
    dev_data_path = '../datasets/AAPD/dev.tsv'

    trainset = preprocess.AAPDDataset(train_data_path)
    validset = preprocess.AAPDDataset(dev_data_path)

    # src_vocab = data['dict']['src']
    # tgt_vocab = data['dict']['tgt']
    # config.src_vocab_size = src_vocab.size()
    # config.tgt_vocab_size = tgt_vocab.size()

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=lambda b: preprocess.process_batch_input(b, opt))
    if hasattr(opt, 'valid_batch_size'):
        valid_batch_size = opt.valid_batch_size
    else:
        valid_batch_size = opt.batch_size
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=valid_batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=lambda b: preprocess.process_batch_input(b, opt))

    return {'trainset': trainset, 'validset': validset,
            'trainloader': trainloader, 'validloader': validloader, }
    # 'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}

# Define the KNN module
class KNN(nn.Module):
    def __init__(self, input_dim, output_dim, k=5, tau=1.0):
        super(KNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.tau = tau
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, query, database, database_labels):
        # Calculate KNN predictions
        distances = torch.cdist(query, database, p=2)
        _, indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        knn_labels = torch.gather(database_labels, dim=0, index=indices.unsqueeze(2).expand(-1, -1, self.output_dim))
        knn_predictions = knn_labels.mean(dim=1)
        return knn_predictions

# Define the contrastive learning objective
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, text_embeddings, text_labels):
        # Calculate the contrastive loss
        similarity_matrix = torch.matmul(text_labels, text_labels.t())
        similarity_matrix = similarity_matrix / self.tau
        contrastive_loss = nn.CrossEntropyLoss()(similarity_matrix, torch.arange(len(text_embeddings)).to(text_embeddings.device))
        return contrastive_loss


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    # print(score_mat)
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)


def Ndcg_k(true_mat, score_mat, k):
    def get_factor(label_count, k):
        res = []
        for i in range(len(label_count)):
            n = int(min(label_count[i], k))
            f = 0.0
            for j in range(1, n + 1):
                f += 1 / np.log(j + 1)
            res.append(f)
        return np.array(res)

    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)


def build_model():
    # for k, v in config.items():
    #     print_log("%s:\t%s\n" % (str(k), str(v)))

    # model
    print('building model...\n')
    # model = getattr(models, opt.model)(config)
    # if checkpoints is not None:
    #     model.load_state_dict(checkpoints['model'])
    # if opt.pretrain:
    #     print('loading checkpoint from %s' % opt.pretrain)
    #     pre_ckpt = torch.load(opt.pretrain)['model']
    #     pre_ckpt = OrderedDict({key[8:]: pre_ckpt[key] for key in pre_ckpt if key.startswith('encoder')})
    #     print(model.encoder.state_dict().keys())
    #     print(pre_ckpt.keys())
    #     model.encoder.load_state_dict(pre_ckpt)
    # if use_cuda:
    #     model.cuda()

    # # optimizer
    # if checkpoints is not None:
    #     optim = checkpoints['optim']
    # else:
    #     optim = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
    #                          lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    # optim.set_parameters(model.parameters())
    embedding_size = 300
    hidden_size = 512
    label_size = 54
    batch_size = opt.batch_size

    model = LDGN(batch_size, opt.max_seq_length, embedding_size, hidden_size, label_size, opt.embedding)
    # outputs = model.forward(torch.tensor(A), lstm_out)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model, optim


def train_model(model, data, optim, epochs, A):
    model.train()
    if use_cuda:
        model.to(device)
    trainloader = data['trainloader']
    devloader = data['validloader']

    write_after = 1000 // opt.batch_size
    counter = 0
    loss = 0
    predictions_k = KNeighborsClassifier(n_neighbors=5)

    for epoch in range(1, epochs + 1):
        print('Epoch %d' % epoch)
        mic_f1 = []
        for contexts, labels in tqdm(trainloader, total=len(trainloader)):
            model.train()
            contexts = contexts.to(device)
            A = A.to(device)
            model = model.to(device)

            output = model(A, contexts)
            targets = torch.tensor([[float(label == 1) for label in batch_label] for batch_label in labels],
                                   dtype=torch.float32).to(device)
            predictions_k.fit(contexts.detach().cpu().float(), labels.detach().cpu().float())
            loss_fn = torch.nn.BCELoss(reduction='sum')

            # loss = loss_fn(output, targets.view(*output.shape))
            # 总和BCELOSS
            bce_loss = loss_fn(output, targets.view(*output.shape))
            CL = Contrastive_loss(labels=labels, contexts=contexts)
            CLlose = CL.Get_Contrastiveloss()

            loss = bce_loss + 0.1 * CLlose
            optim.zero_grad()
            loss.backward()
            optim.step()

            # prec = precision_k(targets.view(1, 54).cpu().numpy(), output.view(1, 54).detach().cpu().numpy(), 5)
            # prec_k.append(prec)
            # loss = 0
            counter += 1
            if counter % write_after == 0:
                # print(f'Prec@k:{np.array(prec_k).mean(axis=0)}')
                prec_k = []
                model.eval()
                eval_loss = 0
                count = 0
                eval_precision = []
                ndcg_k = []
                predictions, true_labels = [], []
                f111=[]
                acc111=[]
                for contexts, labels in devloader:
                    # print(contexts.shape)
                    contexts = contexts.to(device)
                    model = model.to(device)
                    pred = model(A, contexts)
                    # print(len(pred))
                    # preds = pred.logit.sigmoid().round()  # 用于多标签分类的阈值决策
                    # print(len(preds))
                    # for i in pred:
                    #     a = torch.logit(i)
                    #     b = a.sigmoid().round()
                    #     print(b)
                    #     break
                    k_pred = predictions_k.predict(contexts.cpu().numpy())
                    targets = np.array([[float(label == 1) for label in batched_label] for batched_label in labels])
                    preds = pred.detach().cpu().float().view(*targets.shape)
                    # for batch_label in labels:
                    #     tl = []
                    #     for pl in batch_label:
                    #        tl.append(float(pl == 1))
                    #     true_labels.extend(tl)
                    #     print("tl:  ", tl)
                    for i in range(len(preds)):
                        templ = torch.logit(preds[i])
                        l = templ.sigmoid().round()
                        tl=[]
                        for pl in labels[i]:
                            tl.append(float(pl == 1))
                        true_labels.append(tl)
                        predictions.append((1 * l.cpu().numpy()+ 0 * k_pred[i]).round())
                        # f1 = f1_score(tl_1, l_1, average='micro', zero_division=1)
                        # print(f'F1:{f1}')
                        # acc = accuracy_score(tl_1, l_1)
                        # print(f'acc:{acc}')
                    # for batch_pred in preds:
                    #     templ = torch.logit(batch_pred)
                    #     l = templ.sigmoid().round()
                    #     predictions.extend(l.cpu().numpy())
                    # predictions = np.array(predictions)
                    # true_labels = np.array(true_labels)
                    # #print(len(predictions), "::",len(true_labels))
                    # f1 = f1_score(true_labels, predictions, average='micro', zero_division=1)
                    # print(f'F1:{f1}')
                    # acc = accuracy_score(true_labels, predictions)
                    # print(f'acc:{acc}')

                    # print(targets.shape, pred.shape)
                    #eval_precision.append(precision_k(targets, pred.detach().cpu().float().view(*targets.shape), 5))
                    # print(precision_k(targets, pred.detach().cpu().float(), 5))
                    eval_loss += loss_fn(pred.detach().cpu().float(),
                                         torch.tensor(targets, dtype=torch.float32).view(*pred.shape))
                    #ndcg = Ndcg_k(targets, pred.detach().cpu().float().view(*targets.shape), 5)
                    #ndcg_k.append(ndcg)

                    count += 1

                f1 = f1_score(true_labels, predictions, average='micro', zero_division=1)
                # print(f'F1:{f1}')
                mic_f1.append(f1)
                # acc = accuracy_score(true_labels, predictions)
                # print(f'acc:{acc}')
        print(f'micro-f1:{np.mean(mic_f1, axis=0)}')
                # print(f'F1:{np.mean(f111, axis=0)}')
                # print(f'Acc:{np.mean(acc111, axis=0)}')
               # print(f'P@k:{np.mean(eval_precision, axis=0)}')
                #print(f'DCG@k:{np.mean(ndcg_k, axis=0)}')

            # print(f'Eval loss:{eval_loss / (count)}')

        # utils.progress_bar(params['updates'], config.eval_interval)
        # params['updates'] += 1

        # if params['updates'] % config.eval_interval == 0:
        #     params['log']("epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
        #                   % (epoch, params['report_loss'], time.time()-params['report_time'],
        #                      params['updates'], params['report_correct'] * 100.0 / params['report_total']))
        #     print('evaluating after %d updates...\r' % params['updates'])
        #     score = eval_model(model, data, params)
        #     for metric in config.metrics:
        #         params[metric].append(score[metric])
        #         if score[metric] >= max(params[metric]):
        #             with codecs.open(params['log_path']+'best_'+metric+'_prediction.txt','w','utf-8') as f:
        #                 f.write(codecs.open(params['log_path']+'candidate.txt','r','utf-8').read())
        #             save_model(params['log_path']+'best_'+metric+'_checkpoint.pt', model, optim, params['updates'])
        #     model.train()
        #     params['report_loss'], params['report_time'] = 0, time.time()
        #     params['report_correct'], params['report_total'] = 0, 0

        # if params['updates'] % config.save_interval == 0:
        #     save_model(params['log_path']+'checkpoint.pt', model, optim, params['updates'])

    # optim.updateLearningRate(score=0, epoch=epoch)


# def eval_model(model, data, params):

#     model.eval()
#     reference, candidate, source, alignments = [], [], [], []
#     count, total_count = 0, len(data['validset'])
#     validloader = data['validloader']
#     tgt_vocab = data['tgt_vocab']

#     for src, tgt, src_len, tgt_len, original_src, original_tgt in validloader:

#         if config.use_cuda:
#             src = src.cuda()
#             src_len = src_len.cuda()

#         with torch.no_grad():
#             if config.beam_size > 1 and (not config.global_emb):
#                 samples, alignment, _ = model.beam_sample(src, src_len, beam_size=config.beam_size, eval_=True)
#             else:
#                 samples, alignment = model.sample(src, src_len)

#         candidate += [tgt_vocab.convertToLabels(s.tolist(), utils.EOS) for s in samples]
#         source += original_src
#         reference += original_tgt
#         if alignment is not None:
#             alignments += [align for align in alignment]

#         count += len(original_src)
#         utils.progress_bar(count, total_count)

#     if config.unk and config.attention != 'None':
#         cands = []
#         for s, c, align in zip(source, candidate, alignments):
#             cand = []
#             for word, idx in zip(c, align):
#                 if word == utils.UNK_WORD and idx < len(s):
#                     try:
#                         cand.append(s[idx])
#                     except:
#                         cand.append(word)
#                         print("%d %d\n" % (len(s), idx))
#                 else:
#                     cand.append(word)
#             cands.append(cand)
#             if len(cand) == 0:
#                 print('Error!')
#         candidate = cands

#     with codecs.open(params['log_path']+'candidate.txt','w+','utf-8') as f:
#         for i in range(len(candidate)):
#             f.write(" ".join(candidate[i])+'\n')

#     results = utils.eval_metrics(reference, candidate, label_dict, params['log_path'])
#     score = {}
#     result_line = ""
#     for metric in config.metrics:
#         score[metric] = results[metric]
#         result_line += metric + ": %s " % str(score[metric])
#     result_line += '\n'

#     params['log'](result_line)

#     return score


# def save_model(path, model, optim, updates):
#     model_state_dict = model.state_dict()
#     checkpoints = {
#         'model': model_state_dict,
#         'config': config,
#         'optim': optim,
#         'updates': updates}
#     torch.save(checkpoints, path)


# def build_log():
#     # log
#     if not os.path.exists(config.logF):
#         os.makedirs(config.logF)
#     if opt.log == '':
#         log_path = config.logF + str(int(time.time() * 1000)) + '/'
#     else:
#         log_path = config.logF + opt.log + '/'
#     if not os.path.exists(log_path):
#         os.makedirs(log_path)
#     print_log = utils.print_log(log_path + 'log.txt')
#     return print_log, log_path


# def main():
#     # checkpoint
#     if opt.restore:
#         print('loading checkpoint...\n')
#         checkpoints = torch.load(opt.restore)
#     else:
#         checkpoints = None

#     data = load_data()
#     print_log, log_path = build_log()
#     model, optim, print_log = build_model(checkpoints, print_log)
#     # scheduler
#     if config.schedule:
#         scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
#     params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
#               'report_correct': 0, 'report_time': time.time(),
#               'log': print_log, 'log_path': log_path}

#     # for metric in config.metrics:
#     #     params[metric] = []
#     # if opt.restore:
#     #     params['updates'] = checkpoints['updates']

#     if opt.mode == "train":
#         for i in range(1, config.epoch + 1):
#             if config.schedule:
#                 scheduler.step()
#                 print("Decaying learning rate to %g" % scheduler.get_lr()[0])
#             train_model(model, data, optim, i, params)
#         for metric in config.metrics:
#             print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))
#     else:
#         score = eval_model(model, data, params) 

def get_inter_label_prob(labels):
    A = np.zeros((len(labels[0]), len(labels[0])))

    def get_conditional_prob(labels, i, j):
        # TODO further optimize this
        # Returning A_ij which denotes the conditional probability of a sample 
        # belonging to category C_i when it belongs to category C_j
        F_j = sum([1 for label in labels if label[j] == '1'])
        F_ij = sum([1 for label in labels if (label[j] == '1') and (label[i] == '1')])
        return F_ij / F_j

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = get_conditional_prob(labels, i, j)
    return A


if __name__ == '__main__':
    data_dict = load_data()
    labels = pd.read_table('../datasets/AAPD/dev.tsv', sep='\t', names=['Labels', 'context'])['Labels']
    # A = get_inter_label_prob(labels)
    # # with open('A.npy', 'w') as f:
    # np.save('A.npy', A)
    # import sys
    # sys.exit(1)
    A = torch.tensor(get_inter_label_prob(labels))
    A = A.to(device)
    opt.A = A
    model, optim = build_model()

    train_model(model, data_dict, optim, opt.epochs, A)
