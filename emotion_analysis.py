# emotion_analysis.py

import torch
import torch.nn.functional as F
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

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
    labels = ["負向", "正向"]
    return {"text": text, "label": labels[pred_class], "confidence": round(probs[0][pred_class].item(), 4)}

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
    colors = ['green' if r['label'] == '正向' else 'red' for r in results]

    # 繪圖儲存
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, confidences, color=colors)
    for bar, result in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{result['label']} {result['confidence']:.2f}",
                 ha='center', va='bottom', fontsize=8)
    plt.ylim(0, 1.05)
    plt.title("逐句情緒分析（紅：負向，綠：正向）")
    plt.xlabel("句子編號")
    plt.ylabel("模型信心分數")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    image_path = "emotion_plot.png"
    plt.savefig(image_path)
    plt.close()

    # 統計資料
    pos_scores = [r['confidence'] for r in results if r['label'] == '正向']
    neg_scores = [r['confidence'] for r in results if r['label'] == '負向']
    pos_mean = np.mean(pos_scores) if pos_scores else 0
    pos_std = np.std(pos_scores) if pos_scores else 0

    neg_mean = np.mean(neg_scores) if neg_scores else 0
    neg_std = np.std(neg_scores) if neg_scores else 0

    # 情緒變化
    emotion_labels = [r['label'] for r in results]
    transitions = sum(1 for i in range(1, len(emotion_labels)) if emotion_labels[i] != emotion_labels[i - 1])
    count = Counter(emotion_labels)

    stats = {
        "正向平均值": round(pos_mean, 3),
        "正向標準差": round(pos_std, 3),
        "負向平均值": round(neg_mean, 3),
        "負向標準差": round(neg_std, 3),
        "正向句子數": count.get('正向', 0),
        "負向句子數": count.get('負向', 0),
        "正負向變化次數": transitions
    }

    return results, stats, image_path
