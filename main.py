import streamlit as st
st.title("情緒分析網頁")
from emotion_analysis import sentence_split
text = st.text_area("請輸入文字")
output=sentence_split(text)
print(output)
if st.button("開始斷句與情緒分析"):
    st.write("分析中...")
    output = sentence_split(text)
    st.write("分析結果原始輸出：", output)
from emotion_analysis import predict_emotion
from emotion_analysis import analyze_text
results, stats, image_path = analyze_text(text)
st.subheader("逐句情緒分析結果")
for i, r in enumerate(results, 1):
    st.write(f"句子 {i}: 「{r['sentence']}」")
    st.write(f"👉 情緒：{r['label']}（信心值：{r['confidence']:.3f}）")
    st.markdown("---")
st.subheader("情緒統計分析")
st.write(f"🔹 正向句子數：{stats['正向句子數']}")
st.write(f"🔸 負向句子數：{stats['負向句子數']}")
st.write(f"🔁 正負向變化次數：{stats['正負向變化次數']}")

st.write(f"✅ 正向 - 平均值：{stats['正向平均值']:.3f}，標準差：{stats['正向標準差']:.3f}")
st.write(f"⚠️ 負向 - 平均值：{stats['負向平均值']:.3f}，標準差：{stats['負向標準差']:.3f}")

# 顯示圖表
st.subheader("情緒信心圖（紅=負向、綠=正向）")
st.image(image_path, caption="逐句情緒圖", use_column_width=True)