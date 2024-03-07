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

# ステップ4: 評価軸の選択または記入
st.markdown("#### Step4: 評価軸の選択または記入")
st.markdown("※Step4ではキーワードを評価したい軸を設定")
axis_optionsA = ["その他（自由記述）","コスト効率性", "短期成果", "顧客満足度", "イノベーション度", "リスク", "環境への影響"]
axis_optionsB = ["その他（自由記述）","施策効果", "長期成果", "業務効率", "実行の容易さ", "市場の成長性", "社会貢献"]

# x軸の選択
col1, col2 = st.columns(2)
with col1:
    x_axis_selection = st.selectbox("x軸の評価軸を選択してください", options=axis_optionsA)
with col2:
    if x_axis_selection == "その他（自由記述）":
        x_axis_custom = st.text_input("x軸の評価軸（自由記述）")

# y軸の選択（同様に処理）
col3, col4 = st.columns(2)
with col3:
    y_axis_selection = st.selectbox("y軸の評価軸を選択してください", options=axis_optionsB)
with col4:
    if y_axis_selection == "その他（自由記述）":
        y_axis_custom = st.text_input("y軸の評価軸（自由記述）")

# ステップ5: 類似度計算とグラフ表示
st.markdown("##### Step5: 類似度計算とグラフ表示")
st.markdown("ユーザー課題およびデータジャケットから抽出したキーワードの説明をベクトル化し、軸との類似性を評価して２軸にてマッピング")

# 実行ボタン
# ステップ5: 類似度計算とグラフ表示
if st.button('実行'):
    # エクセルファイルの読み込みとキーワードの抽出
    keywords_data = pd.DataFrame()
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            keywords_data = pd.concat([keywords_data, df], ignore_index=True)
    print(keywords_data)

    # BERTモデルのロード
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

    # 評価軸のベクトル化
    if x_axis_selection == "その他（自由記述）":
        x_axis_embedding = get_bert_embedding(x_axis_custom)
    else:
        x_axis_embedding = get_bert_embedding(x_axis_selection)
    if y_axis_selection == "その他（自由記述）":
        y_axis_embedding = get_bert_embedding(y_axis_custom)
    else:
        y_axis_embedding = get_bert_embedding(y_axis_selection)

    # キーワードごとに類似度計算
    keywords_data['x_similarity'] = keywords_data['キーワードの説明'].apply(lambda x: calculate_similarity(get_bert_embedding(x), x_axis_embedding))
    keywords_data['y_similarity'] = keywords_data['キーワードの説明'].apply(lambda x: calculate_similarity(get_bert_embedding(x), y_axis_embedding))

#　＜＜＜類似度は標準化し、０～１０点満点とする。そして軸は５のところに来るようにし、４象限を表現する＞＞＞

    # キーワード単位のグラフ
    fig = go.Figure(data=[go.Scatter(
        x=keywords_data['x_similarity'],
        y=keywords_data['y_similarity'],
        mode='markers+text',
        text=keywords_data['キーワード'],
        marker=dict(size=12, color=keywords_data['x_similarity'], colorscale='Viridis', showscale=True)
    )])

    if x_axis_selection == "その他（自由記述）" and y_axis_selection == "その他（自由記述）":
        fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_custom, yaxis_title=y_axis_custom)
    elif x_axis_selection != "その他（自由記述）" and y_axis_selection == "その他（自由記述）":
        fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_selection, yaxis_title=y_axis_custom)
    elif x_axis_selection == "その他（自由記述）" and y_axis_selection != "その他（自由記述）":
        fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_custom, yaxis_title=y_axis_selection)
    else:
        fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_selection, yaxis_title=y_axis_selection)

    # fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_custom, yaxis_title=y_axis_custom)
    st.plotly_chart(fig, use_container_width=True)

# ＜＜＜キーワード単位のベクトルが出ているため、それらを足し合わせてデータジャケット単位のベクトルを作成し、そのベクトルと２軸のベクトルの類似度を計算。グラフ化する＞＞＞

# ＜＜＜データジャケット単位のベクトル類似度がでているため、それらを表にしてここに表示（st.write？）それをchatGPTに貼り付けることで近しいものをグループ化し、名称をつける。それを貼り付けるテキストボックスを用意し、貼り付けられたらグループ化単位の類似度を計算、グラフ化する＞＞＞
