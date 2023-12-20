import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import json
import faiss

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# （1）加载和预处理数据
f1 = open("./data_AAPR_data/data1", "r")
f2 = open("./data_AAPR_data/data2", "r")
f3 = open("./data_AAPR_data/data3", "r")
f4 = open("./data_AAPR_data/data4", "r")

data1 = json.load(f1)
data2 = json.load(f2)
data3 = json.load(f3)
data4 = json.load(f4)

f1.close()
f2.close()
f3.close()
f4.close()

data = {**data1, **data2, **data3, **data4}

df = pd.DataFrame(data)
col = df.columns
index_ = df.index
data = np.array(df).T
df = pd.DataFrame(data)
df.columns = index_
df.index = col

# 假设category列是多标签的列，将其转换为独热编码
categories = df['category'].str.get_dummies(sep=',')

# 合并数据
df = pd.concat([df, categories], axis=1)
df.drop(['category'], axis=1, inplace=True)


# （2）创建数据集类
class AAPDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,  # 确保超过最大长度的文本被截断
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

# （3）准备训练和验证数据
# 分割数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(df['abstract'], categories.values, test_size=0.1)

# 初始化Tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_path = "./model_save_path"
tokenizer = BertTokenizer.from_pretrained(model_path)
max_len = 512  # 或其他适合的长度

# 创建数据集
train_dataset = AAPDDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = AAPDDataset(val_texts, val_labels, tokenizer, max_len)

# （4）创建数据加载器
batch_size = 32  # 或其他适合的批大小
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

# （5）初始化BERT模型
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories.columns),output_hidden_states=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = model.to(device)

# （8.1）加载预训练模型
# 模型保存路径
model_path = "./model_save_path"

# 加载模型
model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
model = model.to(device)

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# （6）设置优化器和训练参数
epochs = 5  # 或根据需要调整

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * epochs

# 可以选择合适的学习率调度器
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# （7）定义训练和评估函数
def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model = model.train()
    losses = []
    for batch in data_loader:
        optimizer.zero_grad()  # 清零梯度
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    # losses = []
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # loss = outputs.loss
            # losses.append(loss.item())

            preds = outputs.logits.sigmoid().round()  # 用于多标签分类的阈值决策
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # 计算指标
    f1 = f1_score(true_labels, predictions, average='micro')
    acc = accuracy_score(true_labels, predictions)

    return f1, acc

f_1 = open('Bert+KNN_new.log','w')
# （8）训练模型
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')

    print('-' * 10)

    train_loss = train_epoch(model, train_data_loader, optimizer, device)
    print(f'Train loss {train_loss}')

    # val_loss,
    val_f1, val_acc = eval_model(model, val_data_loader, device)
    # print(f'Validation loss {val_loss}')
    print(f'Validation F1 Score: {val_f1}')
    print(f'Validation Accuracy: {val_acc}')
    f_1.write(f'Epoch {epoch + 1}/{epochs}'+'\n')
    f_1.write('-' * 10+'\n')
    f_1.write(f'Train loss {train_loss}'+'\n')
    f_1.write(f'Validation F1 Score: {val_f1}'+'\n')
    f_1.write(f'Validation Accuracy: {val_acc}'+'\n')

# （9）训练结束后保存模型
model_path = "./model_save_path_new"  # 您可以选择一个路径来保存模型
model.save_pretrained(model_path)  # 保存模型的权重和配置
tokenizer.save_pretrained(model_path)  # 保存相应的tokenizer

# （10）加入KNN模型
# 在训练结束后构建数据存储
model = model.eval()
datastore = []

with torch.no_grad():
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].numpy()  # 获取原始标签

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # representations = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # 取CLS标记的输出作为表示
        representations = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # 取最后一层的CLS标记的输出作为表示

        for rep, label in zip(representations, labels):
            datastore.append((rep, label))

# 使用Faiss创建索引
# 假设datastore_representations是一个numpy数组，其中包含了所有文本的BERT表示
datastore_representations = np.array([x[0] for x in datastore]).astype(np.float32)
datastore_labels = [np.array(x[1]) for x in datastore]

print("Datastore label type:", type(datastore_labels[0]))  # 打印第一个元素的类型
print("Datastore label shape:", datastore_labels[0].shape)  # 打印第一个元素的形状

# 创建一个Faiss索引
index = faiss.IndexFlatL2(datastore_representations.shape[1])  # L2距离
index.add(datastore_representations)  # 向索引中添加表示

# KNN进行预测的新函数
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

# 评估KNN模型
def final_eval_model(model, data_loader, device, lambda_param):
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 获取BERT模型的输出
            bert_predictions = outputs.logits.sigmoid().cpu().numpy()
            # 获取文本表示
            text_representations = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            # 结合BERT和KNN预测
            combined_predictions = []
            for bert_pred, text_rep in zip(bert_predictions, text_representations):
                knn_pred = knn_predict_with_faiss(index, datastore_labels, text_rep, k=5, tau=1)
                combined_pred = lambda_param * np.array(knn_pred) + (1 - lambda_param) * np.array(bert_pred)
                combined_predictions.append(combined_pred.round())
            # 更新predictions和true_labels
            predictions.extend(combined_predictions)
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
        # 打印predictions和true_labels的形状
    print("predictions shape:", predictions.shape)
    print("true_labels shape:", true_labels.shape)
    print(predictions[0])
    print(true_labels[0])

    # 计算指标
    f1 = f1_score(true_labels, predictions, average='micro')
    acc = accuracy_score(true_labels, predictions)

    return f1, acc

val_f1, val_acc = final_eval_model(model, val_data_loader, device, 0.5)
print(f'KNN & BERT Validation F1 Score with Lambda(0.5): {val_f1}')
print(f'KNN & BERT Validation Accuracy with Lambda(0.5): {val_acc}')
f_1.write(f'KNN & BERT Validation F1 Score with Lambda(0.5): {val_f1}'+'\n')
f_1.write(f'KNN & BERT Validation Accuracy with Lambda(0.5): {val_acc}'+'\n')

val_f1, val_acc = final_eval_model(model, val_data_loader, device, 1)
print(f'KNN & BERT Validation F1 Score with Lambda(1): {val_f1}')
print(f'KNN & BERT Validation Accuracy with Lambda(1): {val_acc}')
f_1.write(f'KNN & BERT Validation F1 Score with Lambda(1): {val_f1}'+'\n')
f_1.write(f'KNN & BERT Validation Accuracy with Lambda(1): {val_acc}'+'\n')

f_1.close()
