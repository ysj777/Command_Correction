import torch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datasets
from datasets import Dataset, DatasetDict
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ast
from transformers import AutoTokenizer,BertTokenizer, BertForTokenClassification, get_scheduler, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModel
from torch.optim import AdamW
import evaluate
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
        api_key="",  # 請填入你的 OpenAI API 金鑰
    )

id2label = {
    0: 'O',
    1: 'B-Modify',
    2: 'B-Filling',
}

label2id = {
    'O': 0,
    'B-Modify': 1,
    'B-Filling': 2,
}

label_list = ['O', 'B-Modify', 'B-Filling']

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx]['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(self.dataset[idx]['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.dataset[idx]['labels'], dtype=torch.long)
        }

def convert_ids_to_labels(pred_ids, id2label):
    return [id2label[i] for i in pred_ids]

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.command
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
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

class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('bert-base-chinese')
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(256, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

class Input_ModelClass(torch.nn.Module):
    def __init__(self):
        super(Input_ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained('bert-base-chinese')
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


def llm_input_classifier(text):

    prompt = """你是一個語音輸入指令分類器，請判斷使用者的語音是以下哪種類型：

    0：一般語音內容（如敘述、聊天內容）

    1：編輯指令（例如「請把某個字改成別的字」、「刪除某句話」）

    2：操作指令（例如「停止錄音」、「送出目前內容」、「重新開始錄音」）

    若屬於操作指令（2），請再細分為下列三種，並輸出對應的代號：

    2-1：停止錄音指令
    表示暫停錄音，不再接收語音辨識內容。
    例句：停止錄音、錄到這邊就好

    2-2：送出目前內容指令
    表示確認、傳送目前的輸入內容。
    例句：送出、傳出去

    2-3：重新開始錄音（繼續說話）指令
    表示使用者要「繼續錄音」，恢復語音辨識。常見於先前已停止錄音後，現在希望繼續。
    例句：重新開始、繼續錄音

    請根據輸入的句子，輸出下列代號其中之一：
    0、1、2-1、2-2 或 2-3"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
        
            
    except Exception as e:
        print(f"Error at index {index}: {e}")

def Check_command_type(command):

    if '刪除' in command:
        return 2
    elif '新增' in command:
        return 1
    elif '改成' in command:
        return 0
    else:
        return None

def cmd_prediction(model, command, tokenizer, device='cpu', max_len=64):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            command,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class

def input_prediction(model, command, tokenizer, device='cpu', max_len=64):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            command,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class

def md_prediction(md_model, text, md_tokenizer, device):
    md_model.eval()
    with torch.no_grad():
        inputs = md_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = md_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
        word_ids = md_tokenizer(text, return_offsets_mapping=True, truncation=True, padding='max_length', max_length=256).word_ids()
        valid_indices = [i for i, word_id in enumerate(word_ids) if word_id is not None]

        filtered_predictions = [predictions[i] for i in valid_indices]

    return filtered_predictions


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Without Error cmd
    #SIGHAN_test_df = pd.read_csv("./data/Sighan_modify_area_test.csv", encoding = 'utf-8')
    #wiki_test_df = pd.read_csv("data/modifyClassifier_test.csv", encoding = 'utf-8')

    # With Error cmd
    SIGHAN_test_df = pd.read_csv("./Error_cmd_SIGHAN/only_sighan_md_test.csv", encoding = 'utf-8')
    wiki_test_df = pd.read_csv("./Error_cmd/only_error_md_test.csv", encoding = 'utf-8')


    SIGHAN_test_df['label'] = SIGHAN_test_df['label'].apply(ast.literal_eval)
    wiki_test_df['label'] = wiki_test_df['label'].apply(ast.literal_eval)


    # Command Classifier model
    cmd_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    cmd_model = ModelClass().to(device)
    #cmd_model = torch.load("cmd_classify_bert_error.pt", map_location=device)
    #cmd_model = torch.load("cmd_classify_bert_error_sighan.pt", map_location=device)
    cmd_model = torch.load("cmd_classify_bert_error_mix.pt", map_location=device)
    #cmd_model = torch.load("cmd_classify_bert_mix.pt", map_location=device)

    input_model = Input_ModelClass().to(device)

    #input_model.load_state_dict(torch.load("./error_input_classify_bert_state.pt", map_location=device))
    #input_model = torch.load("error_input_classify_bert_sighan.pt", map_location=device)
    input_model = torch.load("error_input_classify_bert_mix.pt", map_location=device)
    """state_dict = torch.load("Mix_input_classify_bert.pt", map_location=device)
    input_model.load_state_dict(state_dict)"""
    #input_model = torch.load("Mix_input_classify_bert.pt", map_location=device)

    # Command Labeler model
    md_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    md_model= AutoModelForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased", 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    ).to(device)
    #md_model.load_state_dict(torch.load("./my_awesome_wnut_model/best_model_state_error.pt", map_location=device))
    #md_model.load_state_dict(torch.load("./my_awesome_wnut_model/best_model_state_error_sighan.pt", map_location=device))
    md_model.load_state_dict(torch.load("./my_awesome_wnut_model/best_model_state_error_mix.pt", map_location=device))
    #md_model.load_state_dict(torch.load("./my_awesome_wnut_model/best_model_state_mix.pt", map_location=device))

    # Success Rate
    success = []
    results =[]

    for index, row in tqdm(SIGHAN_test_df.iterrows(), total=len(SIGHAN_test_df)):

        text = row["text"]
        #print(text)
        labels = row["label"]  # 如果這是 list 格式的欄位
        parts = text.split(" [SEP] ")
        pre_sep_text = parts[0]
        post_sep_text = parts[1]
        record = {
            "index": index,
            "text": text,
            "pre_sep_text": pre_sep_text,
            "post_sep_text": post_sep_text,
            "llm_input_type_pre": 0,
            "llm_input_type_post": 1,
            "cmd_type_true": None,
            "cmd_type_pred": None,
            "label_true": labels,
            "label_pred": labels,
            "correct": False,
            "error_stage": None  # 記錄錯在哪一關
        }

        text_prediction = input_prediction(input_model, pre_sep_text, cmd_tokenizer, device=device)
        
        if (text_prediction != 0):
            #print("Input Type Error")
            record['llm_input_type_pre'] = 1
            record["error_stage"] = "llm_pre"
            results.append(record)
            success.append(False)
            continue

        command_prediction = input_prediction(input_model, post_sep_text, cmd_tokenizer, device=device)
        
        if (command_prediction != 1):
            #print("Input Type Error")
            record['llm_input_type_pre'] = 0
            record["error_stage"] = "llm_post"
            results.append(record)
            success.append(False)
            continue


        """if (llm_input_classifier(pre_sep_text) != "0"):
            #print("Input Type Error")
            record['llm_input_type_pre'] = 1
            record["error_stage"] = "llm_pre"
            results.append(record)
            success.append(False)
            continue
        if (llm_input_classifier(post_sep_text) != "1"):
            #print("Input Type Error")
            record['llm_input_type_post'] = 0
            record["error_stage"] = "llm_post"
            results.append(record)
            success.append(False)
            continue"""

        # Command Type
        command_type = Check_command_type(post_sep_text)
        record['cmd_type_true'] = command_type
        record['cmd_type_pred'] = command_type
        #print("command_type:", command_type)

        predict_cmd_type = cmd_prediction(cmd_model, post_sep_text, cmd_tokenizer, device=device)

        if (predict_cmd_type != command_type):
            #print("Cmd Type Error")
            record['cmd_type_pred'] = predict_cmd_type
            record["error_stage"] = "cmd_type"
            results.append(record)
            success.append(False)
            continue
        
        predict_label = md_prediction (md_model, text, md_tokenizer, device = device)
        predict_label = convert_ids_to_labels(predict_label, id2label)
        #print("Predicted label", predict_label)

        if predict_label == labels:
            record["correct"] = True
            success.append(True)
            #print("Prediction Correct")
        else:
            success.append(False)
            #print("Prediction Incorrect")
            #print("Truth label", labels)
            record["correct"] = False
            record["error_stage"] = "token_label"
        
        results.append(record)

    results_df = pd.DataFrame(results)
    results_df.to_csv("Wiki_prediction_results.csv", index=False, encoding="utf-8-sig")
    print("結果已儲存至 prediction_results.csv")

    accuracy = results_df["correct"].mean()
    print(f"\n✅ Prediction Accuracy: {accuracy:.2%} ({results_df['correct'].sum()}/{len(results_df)})")
    

if __name__ == '__main__':
    main()