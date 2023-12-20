import torch
import torch.utils.data
import faiss
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
import torch
import torch.nn as nn
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


def knn_predict_with_faiss(index, datastore_labels, text_representation, k=5, tau=1):
    # 使用Faiss找到最近的k个邻居及其距离
    distances, indices = index.search(np.array([text_representation]).astype(np.float32), k)

    # 归一化距离（将距离映射到0到1的范围内）
    normalized_distances = (distances[0] - np.min(distances[0])) / (np.max(distances[0]) - np.min(distances[0]))

    # 计算权重（使用指数衰减来减少距离的影响）
    weights = np.exp(-normalized_distances / tau)
    weights /= np.sum(weights)

    # 聚合KNN预测
    N = len(datastore_labels[0])  # 假设所有标签向量长度相同
    knn_prediction = np.zeros(N)

    for idx, weight in zip(indices[0], weights):
        knn_prediction += weight * datastore_labels[idx]

    return knn_prediction


# Define the contrastive learning objective
class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau

    def forward(self, text_embeddings, text_labels):
        # Calculate the contrastive loss
        similarity_matrix = torch.matmul(text_labels, text_labels.t())
        similarity_matrix = similarity_matrix / self.tau
        contrastive_loss = nn.CrossEntropyLoss()(similarity_matrix,
                                                 torch.arange(len(text_embeddings)).to(text_embeddings.device))
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
    print('building model...\n')
    embedding_size = 300
    hidden_size = 512
    label_size = 54
    batch_size = opt.batch_size

    model = LDGN(batch_size, opt.max_seq_length, embedding_size, hidden_size, label_size, opt.embedding)
    # outputs = model.forward(torch.tensor(A), lstm_out)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model, optim


def train_model(model, data, optim, epochs, A, faissindex):
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
                ldgnpredictions, true_labels = [], []
                f111 = []
                acc111 = []
                combined_predictions = []
                for contexts, labels in devloader:
                    # print(contexts.shape)
                    contexts = contexts.to(device)
                    model = model.to(device)
                    pred = model(A, contexts)
                    # k_pred = predictions_k.predict(contexts.cpu().numpy())
                    targets = np.array([[float(label == 1) for label in batched_label] for batched_label in labels])
                    preds = pred.detach().cpu().float().view(*targets.shape)
                    for i in range(len(preds)):
                        templ = torch.logit(preds[i])
                        l = templ.sigmoid().round()
                        tl = []
                        for pl in labels[i]:
                            tl.append(float(pl == 1))
                        true_labels.append(tl)
                        ldgnpredictions.append(l.cpu().numpy().round())
                    # text_representations = contexts.cpu().numpy()
                    # for bert_pred, text_rep in zip(ldgnpredictions, text_representations):
                    #     knn_pred = knn_predict_with_faiss(faissindex, datastore_labels, text_rep, k=3, tau=1)
                    #     knn_pred = np.nan_to_num(knn_pred)
                    #     combined_pred = 0.5 * np.array(knn_pred) + (1 - 0.5) * np.array(bert_pred)
                    #     combined_predictions.append(combined_pred.round())
                    eval_loss += loss_fn(pred.detach().cpu().float(),
                                         torch.tensor(targets, dtype=torch.float32).view(*pred.shape))
                    count += 1

                f1 = f1_score(true_labels, combined_predictions, average='micro', zero_division=1)
                # print(f'F1:{f1}')
                mic_f1.append(f1)
        print(f'micro-f1:{np.mean(mic_f1, axis=0)}')


def get_inter_label_prob(labels):
    A = np.zeros((len(labels[0]), len(labels[0])))

    def get_conditional_prob(labels, i, j):
        F_j = sum([1 for label in labels if label[j] == '1'])
        F_ij = sum([1 for label in labels if (label[j] == '1') and (label[i] == '1')])
        return F_ij / F_j

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = get_conditional_prob(labels, i, j)
    return A


if __name__ == '__main__':
    data_dict = load_data()
    train_data_loader = data_dict["trainloader"]
    faissmodel, faissoptim = build_model()
    faissmodel = faissmodel.to(device)
    datastore = []
    with torch.no_grad():
        for faisscontexts, faisslabels in train_data_loader:
            faisscontexts = faisscontexts.to(device).cpu().numpy()
            faisslabels = faisslabels.to(device).cpu().numpy()
            for faissrep, faisslabel in zip(faisscontexts, faisslabels):
                datastore.append((faissrep, faisslabel))
    # 使用Faiss创建索引
    # 假设datastore_representations是一个numpy数组，其中包含了所有文本的BERT表示
    datastore_representations = np.array([x[0] for x in datastore]).astype(np.float32)
    datastore_labels = [np.array(x[1]) for x in datastore]

    print("Datastore label type:", type(datastore_labels[0]))  # 打印第一个元素的类型
    print("Datastore label shape:", datastore_labels[0].shape)  # 打印第一个元素的形状

    # 创建一个Faiss索引
    index = faiss.IndexFlatL2(datastore_representations.shape[1])  # L2距离
    index.add(datastore_representations)  # 向索引中添加表示

    labels = pd.read_table('../datasets/AAPD/dev.tsv', sep='\t', names=['Labels', 'context'])['Labels']
    A = torch.tensor(get_inter_label_prob(labels))
    A = A.to(device)
    opt.A = A
    model, optim = build_model()

    train_model(model, data_dict, optim, opt.epochs, A, index)
