import torch
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 換成 jackietung 模型
MODEL_NAME = "jackietung/bert-base-chinese-finetuned-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 自動讀取模型 labels 數量
label_map = {0: "負向", 1: "中性", 2: "正向"}

def sentence_split(text):
    punctuation = r'[，。！？]'
    sentences = re.split(punctuation, text)
    return [s.strip() for s in sentences if s.strip()]

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
    return {
        "text": text,
        "label": label_map[pred_class],
        "confidence": round(probs[0][pred_class].item(), 4)
    }

def analyze_text(text):
    sentences = sentence_split(text)
    results = []
    for s in sentences:
        result = predict_emotion(s)
        results.append({
            "sentence": s,
            "label": result["label"],
            "confidence": result["confidence"]
        })

    # 圖表資料
    labels = [f"{i+1}" for i in range(len(results))]
    confidences = [r['confidence'] for r in results]

    # 顏色設定
    color_dict = {"正向": "green", "中性": "gray", "負向": "red"}
    colors = [color_dict[r['label']] for r in results]

    # 繪圖儲存
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, confidences, color=colors)
    for bar, result in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{result['label']} {result['confidence']:.2f}",
                 ha='center', va='bottom', fontsize=8)
    plt.ylim(0, 1.05)
    plt.title("逐句情緒分析（紅：負向，灰：中性，綠：正向）")
    plt.xlabel("句子編號")
    plt.ylabel("模型信心分數")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    image_path = "emotion_plot.png"
    plt.savefig(image_path)
    plt.close()

    # 統計資料
    stats = {}
    for label in ["正向", "中性", "負向"]:
        scores = [r['confidence'] for r in results if r['label'] == label]
        stats[f"{label}平均值"] = round(np.mean(scores), 3) if scores else 0
        stats[f"{label}標準差"] = round(np.std(scores), 3) if scores else 0
        stats[f"{label}句子數"] = len(scores)

    # 情緒變化次數
    emotion_labels = [r['label'] for r in results]
    transitions = sum(1 for i in range(1, len(emotion_labels)) if emotion_labels[i] != emotion_labels[i - 1])
    stats["情緒變化次數"] = transitions

    return results, stats, image_path
