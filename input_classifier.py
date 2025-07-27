#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import warnings
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import logging
import matplotlib.pyplot as plt
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


# In[ ]:


torch.cuda.set_device(0)  # 設置為第一個 GPU
torch.cuda.empty_cache()  # 清空 GPU 緩存
print(torch.cuda.memory_allocated())  # 當前已分配的記憶體
print(torch.cuda.memory_reserved())   # 當前已保留的記憶體





# In[ ]:


#trainPath = 'data/inputClassifier_train.csv'
#trainPath = 'Error_cmd/error_input_df_train.csv'
#trainPath = 'Error_cmd_SIGHAN/sighan_input_train.csv'
trainPath = 'Error_cmd_Mix/mix_input_train.csv'
#testPath = 'data/inputClassifier_test.csv'
#testPath= 'Error_cmd/error_input_df_test.csv'
#testPath= 'Error_cmd_SIGHAN/sighan_input_test.csv'
testPath= 'Error_cmd_Mix/mix_input_test.csv'
#testPath= 'Error_cmd_Mix/mix_input_test.csv'
#testPath_1 = 'data/Sighan_input_classifer_test.csv'
train_data = pd.read_csv(trainPath)
test_data = pd.read_csv(testPath)
#test_data_1 = pd.read_csv(testPath_1)
valid_data = train_data.iloc[:len(train_data)//10]
train_data = train_data.iloc[len(train_data)//10:]
#train_data = train_data.sample(frac=0.5, random_state=42)


# In[ ]:


MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-05
MODEL_NAME = "bert-base-chinese"

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# In[ ]:


class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,       # 自動加入特殊標記（如 [CLS], [SEP]）
            max_length=self.max_len,       # 設定最大長度
            truncation=True,               # 顯式啟用截斷
            padding='max_length',          # 使用 'max_length' 模式填充
            return_token_type_ids=True     # 返回 token 類型 ID
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }


# In[ ]:


print("TRAIN Dataset: {}".format(train_data.shape))
print("VALID Dataset: {}".format(valid_data.shape))
print("TEST Dataset: {}".format(test_data.shape))

training_set = SentimentData(train_data, tokenizer, MAX_LEN)
valid_set = SentimentData(valid_data, tokenizer, MAX_LEN)
testing_set = SentimentData(test_data, tokenizer, MAX_LEN)
#testing_set_1 = SentimentData(test_data_1, tokenizer, MAX_LEN) 


# In[ ]:


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **valid_params)
testing_loader = DataLoader(testing_set, **test_params)
#testing_loader_1 = DataLoader(testing_set_1, **test_params)
print("done")


# In[ ]:


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(MODEL_NAME)
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = ModelClass()
model.to(device)


# In[ ]:


loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct


# In[ ]:


def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    # 添加進度條，指定 total 長度
    progress_bar = tqdm(enumerate(training_loader), total=len(training_loader), desc=f"Epoch {epoch}")

    for _, data in progress_bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新進度條資訊
        progress_bar.set_postfix(loss=loss.item(), accuracy=(n_correct * 100) / nb_tr_examples)

    # 結束後打印摘要
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")


# In[ ]:


from tqdm import tqdm

def valid(model, valid_loader):
    model.eval()
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    # 使用 torch.no_grad() 節省內存
    with torch.no_grad():
        # 添加進度條，設定 total 為驗證資料集長度
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating")

        for _, data in progress_bar:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            # 模型預測
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()

            # 計算預測準確性
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            # 更新進度條
            progress_bar.set_postfix(loss=loss.item(), accuracy=(n_correct * 100) / nb_tr_examples)

    # 計算最終的驗證損失與準確率
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu, epoch_loss


# In[ ]:


EPOCHS = 5
#output_model_file = 'error_input_classify_bert.pt'
#output_model_file = 'error_input_classify_bert_mix.pt'
output_model_file = 'Mix_input_classify_bert.pt'
model_to_save = model
model_acc = 0
model_epoch = 0
model_loss = 100

for epoch in range(EPOCHS):
    train(epoch)
    acc, loss = valid(model, valid_loader)
    print("Accuracy on valid data = %0.2f%%" % acc)
    torch.save(model, f'model/bert_{epoch}.pt')
    if acc >= model_acc and loss <= model_loss:
        model_to_save = model
        model_acc = acc
        model_epoch = epoch
        model_loss = loss
        print(f"{epoch} model is better")

torch.save(model_to_save, output_model_file)
print(f'All files saved. {model_epoch} model is best')


# In[ ]:


torch.save(model_to_save, output_model_file)
print(f'All files saved. {model_epoch} model is best')


# In[ ]:


def predict_my(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    output_list = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            # loss = loss_function(outputs, targets)
            # tr_loss += loss.item()
            big_val = torch.argmax(outputs.data)
            # print(big_val.item())
            #n_correct += calcuate_accuracy(big_idx, targets)
            output_list.append(big_val.item())

            # nb_tr_steps += 1
            # nb_tr_examples+=targets.size(0)
            
    # epoch_loss = tr_loss/nb_tr_steps
    # epoch_accu = (n_correct*100)/nb_tr_examples
    # print(f"Validation Loss Epoch: {epoch_loss}")
    # print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return output_list


# In[ ]:


#pre_model = torch.load("error_input_classify_bert_sighan.pt").to(device)
pre_model = ModelClass()
pre_model.load_state_dict(torch.load("Mix_input_classify_bert.pt"))
pre_model.to(device)
#pre_model = torch.load("Mix_input_classify_bert.pt").to(device)

#torch.save(pre_model.state_dict(), 'error_input_classify_bert_state.pt')
#torch.save(pre_model.state_dict(), 'Mix_input_classify_bert.pt')


# In[ ]:


#pre_model = torch.load("error_input_classify_bert.pt").to(device)
#pre_model = torch.load("Mix_input_classify_bert.pt").to(device)
output_list = predict_my(pre_model, testing_loader)
# output_list = predict_my(model_to_save, testing_loader)
test_data['predict'] = pd.Series(output_list)
test_data.to_csv("input_classify_bert_mix.csv", index=False)
print("done")


# In[ ]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, precision_score, f1_score, recall_score

df = test_data
precision = precision_score(df['label'], df['predict'], average='macro')
print(f'Macro precision: {precision}')

precision = precision_score(df['label'], df['predict'], average='micro')
print(f'Precision: {precision}')

cm = confusion_matrix(df['label'], df['predict'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['text', 'cmd'])
disp.plot()
plt.savefig("input_classify_bert_wiki.png")

f1_macro = f1_score(df['label'], df['predict'], average='macro')
f1_micro = f1_score(df['label'], df['predict'], average='micro')
recall_macro = recall_score(df['label'], df['predict'], average='macro')
recall_micro = recall_score(df['label'], df['predict'], average='micro')
print(f"f1 macro: {f1_macro}")
print(f"f1 micro: {f1_micro}")
print(f"recall macro: {recall_macro}")
print(f"recall micro: {recall_micro}")

plt.show()
