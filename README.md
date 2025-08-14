# Command Correction

## API Github (延伸)

* [Command Correction API](https://github.com/ysj777/Command_Correction_API)

## 安裝套件
```
pip install -r requirements.txt 
```
## 資料下載

* Link: [資料集](https://drive.google.com/drive/folders/1bhZi-3dB8DjRewzh7gcV6JUKkGoVXboP?usp=drive_link)
    * 主要訓練資料集: [SIGHAN & Wiki](https://drive.google.com/drive/folders/1XnY_5AvRD7UOYvN9L30o0hACTBUepMcn?usp=drive_link)
    * 經過語音辨識的指令資料: [Error Command](https://drive.google.com/drive/folders/1HJM6xA-o77E6c98RM2Hp4uPPiUwCYxmo?usp=drive_link)

## File Structure
```
├── input_classifier.py # 輸入分類器訓練
├── command_classifier.py # 指令分類器訓練
├── command_labeler.py # 指令標註器訓練
├── system.py # 整體流程效能測試 (Accuracy)
├── LLM.ipynb # 大型語言模型效能測試
├── requirements.txt # 套件需求列表
├── README.md # 使用說明 
```

## 執行主程式

```
python xxx.py
```
