import math

from model.ldgn import LDGN
import sklearn
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import torch
from scipy.spatial import distance

temperature_1 = 10

Total_instance = [{'label': 1, 'content': [2, 1]},
                  {'label': 2, 'content': [3, 1]},
                  {'label': 5, 'content': [1, 2]}]
contexts = [[2, 1], [2, 2], [3, 1]]
labels = [[1, 1], [1, 2], [1, 3]]


class Contrastive_loss():
    def __init__(self, contexts, labels):
        self.contexts = contexts
        self.labels = labels
        self.Euc_Distance = np.zeros((len(self.contexts), len(self.contexts)))
    def Get_Contrastiveloss(self):
        self.Eucli_distancei_total()
        result = 0
        for i in range(len(self.labels)):
            for j in range(i + 1, len(self.labels)):
                result += self.Constrastive_lossij(i, j, i, j)
        return result *2

    def GetC_ij(self, Label1, Label2):
        # Get C_ijEucli_distancei_total
        # L1 = torch.tensor(Label1)
        # L2 = torch.tensor(Label2)
        C_ij = self.labels[Label1]@self.labels[Label2]
        return C_ij

    # batch 1个批次30个句子，40个句子
    # 30000个句子 dataset
    # batch 1个30个句子
    def Dynamic_coefficient(self, labeli, labelj):
        C_ij = self.labels[labeli]@self.labels[labelj]
        Sum_C = 0
        Sum_eucli = 0
        for label in range(len(self.labels)):
            if labeli == label:
                continue
            temp1 = self.labels[labeli]@self.labels[labelj]
            temp2 = self.Euc_Distance[labeli, label]

            Sum_C += temp1
            Sum_eucli += temp2

        return C_ij/Sum_C, Sum_eucli

    def Eucli_distanceij(self, Texti, Textj):
        """Texti should be the vector"""
        Length_dist = 0
        # Text_i = Texti.clone()
        # Text_j = Textj.clone()
        # if len(Text_i) > len(Text_j):
        #     for i in range(len(Text_i) - len(Text_j)):
        #         Text_j.append(0)
        # else:
        #     for i in range(len(Text_j) - len(Text_i)):
        #         Text_i.append(0)
        pdist = torch.nn.PairwiseDistance(p=2)
        d = torch.sum(pdist(self.contexts[Texti], self.contexts[Textj])).item()
        res = np.exp(-d / temperature_1)
        return res

    def Eucli_distancei_total(self):
        for context1 in range(len(self.contexts)):
            for context2 in range(context1 + 1, len(self.contexts)):
                value_distance = self.Eucli_distanceij(context1, context2)
                self.Euc_Distance[context2][context1] = value_distance
                self.Euc_Distance[context1][context2] = value_distance


    def Constrastive_lossij(self, labeli, labelj, contexti, contextj):
        D_coeff, b= self.Dynamic_coefficient(labeli, labelj)
        Loss_ij = - D_coeff * np.log(np.divide( self.Eucli_distanceij(contexti, contextj), b) ) # self.Eucli_distancei_total(Instancei)
        return Loss_ij

