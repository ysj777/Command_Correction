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
from transformers import AutoTokenizer,BertTokenizer, BertForTokenClassification, get_scheduler, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.optim import AdamW
import evaluate
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader


seqeval = evaluate.load("seqeval")

id2label = {
    0: 'O',
    1: 'B-Modify',
    2:'B-Filling'
}

label2id = {
    'O': 0,
    'B-Modify': 1,
    'B-Filling': 2,
}

label_list = ['O', 'B-Modify', 'B-Filling']

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3, id2label=id2label, label2id=label2id)
device = torch.device("cuda")
print(f"Using device: {device}")
model.to(device)

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

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def transform_labels(labels):
    return [label2id.get(label, 0) for label in labels]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def create_dataset(data, is_test=False):
    dataset = []
    for _, row in data.iterrows():
        text = row['text']
        label = row['label']
        if not is_test and len(label) != len(tokenizer.tokenize(text)):
            print(f"Label and text length mismatch in row: {row}")
            continue  # 跳過長度不匹配的數據

        tokenized_inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256)
        if not is_test:
            label = transform_labels(label)
            word_ids = tokenized_inputs.word_ids()

            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            tokenized_inputs['labels'] = label_ids

        dataset.append({
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'labels': tokenized_inputs.get('labels', []) if not is_test else []
        })

    return dataset


def evaluate_test_data(model, tokenizer, test_dataloader, label_list):
    # 生成预测和标签
    true_predictions = []
    true_labels = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # 提取預測
            predictions = torch.argmax(logits, dim=-1)
            batch_predictions = [
                [label_list[p] for p, l in zip(pred, label) if l != -100]
                for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy())
            ]
            batch_labels = [
                [label_list[l] for p, l in zip(pred, label) if l != -100]
                for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy())
            ]
            true_predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

    # 调整序列长度
    # true_predictions = [p[:len(l)] for p, l in zip(true_predictions, true_labels)]
    # true_labels = [l[:len(p)] for p, l in zip(true_predictions, true_labels)]

    # 检查长度是否一致
    assert all(len(p) == len(l) for p, l in zip(true_predictions, true_labels)), "预测序列和真实标签长度不一致！"
    # 计算评估指标
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)



def main():
    set_seed(42)
    #df = pd.read_csv("final_modify_area.csv", encoding="utf-8")
    #train_df = pd.read_csv('./Error_cmd/error_md_df_train.csv', encoding='utf-8')
    train_df = pd.read_csv('./Error_cmd_Mix/mix_md_train.csv', encoding='utf-8')
    #train_df = pd.read_csv('./data/mix_md_classifier_train.csv', encoding='utf-8')
    train_df['label'] = train_df['label'].apply(ast.literal_eval)
    #test_data = pd.read_csv('./Error_cmd/error_md_df_test.csv', encoding='utf-8')
    test_data = pd.read_csv('./Error_cmd_Mix/mix_md_test.csv', encoding='utf-8')
    #test_data = pd.read_csv('./data/modifyClassifier_test.csv', encoding='utf-8')
    test_data['label'] = test_data['label'].apply(ast.literal_eval)
    #test_data_1 = pd.read_csv('./data/Sighan_modify_area_test.csv', encoding='utf-8')
    #test_data_1['label'] = test_data_1['label'].apply(ast.literal_eval)
    #df['label'] = df['label'].apply(ast.literal_eval)
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)
    #val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    print(train_data.shape, val_data.shape, test_data.shape)
    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    test_dataset = create_dataset(test_data)
    #test_dataset_1 = create_dataset(test_data_1)
    train_dataset = CustomDataset(train_dataset)
    print(train_dataset[0])
    val_dataset = CustomDataset(val_dataset)
    print(val_dataset[0])
    test_dataset = CustomDataset(test_dataset)
    print(test_dataset[0])
    #test_dataset_1 = CustomDataset(test_dataset_1)
    #print(test_dataset_1[0])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    #test_dataloader_1 = DataLoader(test_dataset_1, batch_size=8)

    training_args = TrainingArguments(
        output_dir="my_awesome_wnut_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    #torch.save(model.state_dict(), "my_awesome_wnut_model/best_model_state_mix.pt")
    #torch.save(model.state_dict(), "my_awesome_wnut_model/best_model_state_error.pt")
    torch.save(model.state_dict(), "my_awesome_wnut_model/best_model_state_error_mix.pt")
    print("State dict saved successfully!")
    evaluate_test_data(model, tokenizer, test_dataloader, label_list)
    #evaluate_test_data(model, tokenizer, test_dataloader_1, label_list)

if __name__ == '__main__':
    main()