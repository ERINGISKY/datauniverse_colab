import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import scipy.stats
import numpy as np

# Streamlitアプリのタイトル設定
st.title('DataUniverse 共同研究サンプル')

# ステップ1: ユーザーの課題を入力
st.markdown("#### Step1: 課題の記入")
user_issue = st.text_area("あなたの課題を記入してください", placeholder="例：社会的使命でもある“持続可能な”事業運営（業務省人化・自動化・高精度化）")

# ステップ1.5: 参考プロンプトの表示
if user_issue:
    st.markdown("##### Step1.5: 参考プロンプト")
    st.markdown("""
    1. 解決策の仮説を得るためのプロンプトの例

    > 課題（{}）に基づいて、可能な解決策を提案してください。

    2. キーワードを得るためのプロンプトの例

    > 上記の解決策に関連するキーワードを、最大10個、,（カンマ）で区切ってそれぞれ１文程度で記述してください。
    ***
    """.format(user_issue))

# ステップ2: 参考プロンプトを用いたキーワードの記入
st.markdown("#### Step2: キーワードの記入")
st.markdown("※Step2はChatGPTを利用することで自動化が可能だが、ここでは手作業により実施")
input_keywords_str = st.text_area("あなたの課題に基づいて、最も関連性の高い社会的に意味のあるキーワードを最大10個、ここに貼り付けてください", placeholder="例：業務省人化,自動化,高精度化")

# ステップ3: データジャケットファイルのアップロード
st.markdown("#### Step3: データジャケットのアップロード")
st.markdown("※Step3はDataUniverseにあるデータジャケットを自動でChatGPTを介しキーワード化することで自動化が可能だが、ここではテスト用のExcelを使用")
uploaded_files = st.file_uploader("データジャケットのExcelファイルを選択してください", type=['xlsx'], accept_multiple_files=True)

＜＜＜軸の選択は２列にして、左側に選択肢、そこが自由記述の場合は右側に記述としてください。＞＞＞

# ステップ4: 評価軸の選択または記入
st.markdown("#### Step4: 評価軸の選択または記入")
st.markdown("※Step4ではキーワードを評価したい軸を設定")
axis_optionsA = ["その他（自由記述）","コスト効率性", "短期成果", "顧客満足度", "イノベーション度","リスク","環境への影響"]
axis_optionsB = ["その他（自由記述）","施策効果", "長期成果", "業務効率", "実行の容易さ", "市場の成長性","社会貢献"]

x_axis_selection = st.selectbox("x軸の評価軸を選択してください", options=axis_optionsA)
x_axis_custom = st.text_input("x軸の評価軸（自由記述）", placeholder="例：SDGsへの対応") if x_axis_selection == "その他（自由記述）" else x_axis_selection
y_axis_selection = st.selectbox("y軸の評価軸を選択してください", options=axis_optionsB)
y_axis_custom = st.text_input("y軸の評価軸（自由記述）", placeholder="例：入手しやすい") if y_axis_selection == "その他（自由記述）" else y_axis_selection

# ステップ5: 類似度計算とグラフ表示
st.markdown("##### Step5: 類似度計算とグラフ表示")
st.markdown("ユーザー課題およびデータジャケットから抽出したキーワードの説明をベクトル化し、軸との類似性を評価して２軸にてマッピング")

# 実行ボタン
if st.button('実行'):
    # 入力されたキーワードを処理
    input_keywords = input_keywords_str.split(',') if input_keywords_str else []

＜＜＜エクセルファイルの形式を変更しています。アップしているExcelを見て、キーワードの説明部分をベクトル化してください。ただ、グラフにはキーワードを表示してください。また、グラフは複数作成します。ひとつはキーワード単位。ひとつはデータジャケット単位(データジャケットの定量化は紐づくキーワードのベクトルの合計値)、ひとつはデータジャケットの分類単位です（紐づくデータジャケットのベクトルの合計値）。また、キーワードと軸の類似度を計算した後はその類似度に応じてさらにグループ化して表が画面に出るようにしてください。＞＞＞

    # ファイルがアップロードされたら処理
    excel_keywords = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_excel(uploaded_file, engine='openpyxl', header=0)
            # Excelファイルからキーワード列を抽出
            excel_keywords.extend(df['社会的に意味のあるキーワード'].str.split('、').explode().unique().tolist())

    # 入力されたキーワードとExcelからのキーワードを統合
    keywords = list(set(input_keywords + excel_keywords))

    # BERTのトークナイザーとモデルのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # テキストをBERTでベクトル化する関数
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    # 2つのテキスト間のコサイン類似度を計算する関数
    def calculate_similarity(embedding1, embedding2):
        embedding1_np = embedding1.squeeze().numpy()
        embedding2_np = embedding2.squeeze().numpy()
        similarity = 1 - cosine(embedding1_np, embedding2_np)
        return similarity

    # ベクトル化と類似度計算の準備
    x_axis_embedding = get_bert_embedding(x_axis_custom)  # x軸のベクトル化
    y_axis_embedding = get_bert_embedding(y_axis_custom)  # y軸のベクトル化

    # キーワードと評価軸の類似度計算
    x_similarities = []
    y_similarities = []
    for keyword in keywords:
        keyword_embedding = get_bert_embedding(keyword.strip())
        x_sim = calculate_similarity(x_axis_embedding, keyword_embedding)
        y_sim = calculate_similarity(y_axis_embedding, keyword_embedding)
        x_similarities.append(x_sim)
        y_similarities.append(y_sim)

    # 類似度スコアの標準化
    x_similarities_standardized = (np.array(x_similarities) - np.mean(x_similarities)) / np.std(x_similarities)
    y_similarities_standardized = (np.array(y_similarities) - np.mean(y_similarities)) / np.std(y_similarities)

    # 調整されたスコアを使用してグラフにプロット
    fig = go.Figure()
    for i, keyword in enumerate(keywords):
        fig.add_trace(go.Scatter(x=[x_similarities_standardized[i]], y=[y_similarities_standardized[i]],
                                text=keyword, mode='markers+text',
                                textposition='top center'))

    # グラフのレイアウト設定
    fig.update_layout(title="評価軸に基づくキーワードのマッピング",
                    xaxis_title=x_axis_custom,
                    yaxis_title=y_axis_custom,
                    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink'),
                    yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink'))
    st.plotly_chart(fig)
