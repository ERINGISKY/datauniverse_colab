import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import scipy.stats
import numpy as np

# BERTモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# テキストをBERTでベクトル化する関数
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().squeeze()  # CPUに移動してからnumpy変換、さらに1次元化

# 2つのテキスト間のコサイン類似度を計算する関数
def calculate_similarity(embedding1, embedding2):
    # embedding1とembedding2が1次元であることを確認（すでに1次元の想定なので、特に操作は不要）
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# ステップ5の結果を保存する関数
def save_step5_results(fig, fig_dj,df_similarity,df_dj_similarity):
    st.session_state['df_similarity'] = df_similarity
    st.session_state['fig'] = fig
    st.session_state['df_dj_similarity'] = df_dj_similarity
    st.session_state['fig_dj'] = fig_dj

# ステップ5の結果を表示する関数
def display_step5_results():
    if 'df_similarity' in st.session_state:
        st.write("キーワード単位の類似度:")
        st.dataframe(st.session_state['df_similarity'])
    if 'fig' in st.session_state:
        st.plotly_chart(st.session_state['fig'], use_container_width=True)
    if 'df_similarity' in st.session_state:
        st.write("データジャケット単位の類似度:")
        st.dataframe(st.session_state['df_dj_similarity'])
    if 'fig_dj' in st.session_state:
        st.plotly_chart(st.session_state['fig_dj'], use_container_width=True)

# Streamlitアプリのタイトル設定
st.title('DataUniverse 共同研究サンプル')
sumple_text_1 = st.sidebar.markdown("#### Step1の例　コピー用")
sumple_text_2 = st.sidebar.write("社会的使命でもある“持続可能な”事業運営（業務省人化・自動化・高精度化）")
sumple_text_3 = st.sidebar.markdown("#### Step2の例　コピー用")
sumple_text_4 = st.sidebar.write("デジタルツール：業務効率化を実現するオンラインプラットフォームやソフトウェア,プロセス再設計：業務の無駄を削減し効率を向上させるための手順の見直し,RPA：単純作業を自動化するロボットプロセスオートメーションの技術,AI：人間の知能を模倣するコンピューターシステムによる意思決定支援,機械学習：データから学習し予測や分析を行うAIの一分野,データ駆動型アプローチ：データを基にした意思決定で精度を高める戦略,自動テスト：ソフトウェアの品質を保証するためのテストプロセスの自動化,ESG基準：環境、社会、ガバナンスの観点から持続可能性を評価する基準,スキルアップ：従業員の能力向上を目指す教育やトレーニング,テクノロジー統合：複数の新技術を組み合わせて業務の効率化と革新を推進するプロセス業務省人化,自動化,高精度化")

col_step1,col_step15,col_step2 = st.columns(3)
with col_step1:
    # ステップ1: ユーザーの課題を入力
    st.markdown("#### Step1: 課題の記入")
    user_issue = st.text_area("あなたの課題を記入してください", placeholder="例：社会的使命でもある“持続可能な”事業運営（業務省人化・自動化・高精度化）")
with col_step15:
# ステップ1.5: 参考プロンプトの表示
    if user_issue:
        st.markdown("")
        st.markdown("")
        st.markdown("##### Step1.5: 参考プロンプト")
        st.markdown("""
        1. 解決策の仮説を得るためのプロンプトの例

        > 課題（{}）に基づいて、可能な解決策を提案してください。

        2. キーワードを得るためのプロンプトの例

        > 上記の解決策に関連するキーワードを、最大10個、,（カンマ）で区切ってそれぞれ１文程度で記述してください。次のフォーマットで出力してください。＜＜キーワード：キーワードの説明,キーワード：キーワードの説明,キーワード：キーワードの説明,……＞＞
        ***
        """.format(user_issue))
with col_step2:
    # ステップ2: 参考プロンプトを用いたキーワードの記入
    st.markdown("#### Step2: キーワードの記入")
    st.markdown("※Step2はChatGPTを利用することで自動化が可能だが、ここでは手作業により実施")
    input_keywords_str = st.text_area("あなたの課題に基づいて、最も関連性の高い社会的に意味のあるキーワードをここに貼り付けてください", placeholder="例：業務省人化,自動化,高精度化デジタルツール：業務効率化を実現するオンラインプラットフォームやソフトウェア,プロセス再設計：業務の無駄を削減し効率を向上させるための手順の見直し")

# ステップ3: データジャケットファイルのアップロード
st.markdown("#### Step3: データジャケットのアップロード")
st.markdown("※Step3はDataUniverseにあるデータジャケットを自動でChatGPTを介しキーワード化することで自動化が可能だが、ここではテスト用のExcelを使用")
uploaded_files = st.file_uploader("データジャケットのExcelファイルを選択してください", type=['xlsx'], accept_multiple_files=True)
# st.write(uploaded_files)

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
    if input_keywords_str is not None:
        keywords_list = [kw.split("：") for kw in input_keywords_str.split(",")]

        # エラーの原因となる可能性があるデータの形状を確認
        print(keywords_list)

        # 正しい形であればDataFrameの作成を試みる
        try:
            keywords_df = pd.DataFrame(keywords_list, columns=['キーワード', 'キーワードの説明'])
            keywords_df['source'] = 'ユーザー課題'
        except ValueError as e:
            st.write("エラー:", e)
            st.write("keywords_listの形状が不正です。キーワードとその説明のペアが正しくリスト化されているか確認してください。")

    # エクセルファイルの読み込みとキーワードの抽出
    keywords_data = pd.DataFrame()
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            keywords_data = pd.concat([keywords_data, df], ignore_index=True)
            keywords_data['source'] = 'データジャケット'

    # ここで、ステップ2のキーワードDataFrameとステップ3でアップロードされたデータジャケットデータを結合
    combined_df = pd.concat([keywords_df, keywords_data], ignore_index=True)

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

    # 類似度を標準化して0〜10点満点でスケール変換し、グラフの中心を5に設定
    keywords_data['x_similarity'] = (keywords_data['x_similarity'] - keywords_data['x_similarity'].min()) / (keywords_data['x_similarity'].max() - keywords_data['x_similarity'].min()) * 10
    keywords_data['y_similarity'] = (keywords_data['y_similarity'] - keywords_data['y_similarity'].min()) / (keywords_data['y_similarity'].max() - keywords_data['y_similarity'].min()) * 10

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
    # ゼロラインを5に設定して、四象限の中心が5で交わるようにする
    fig.add_shape(type="line", x0=5, y0=0, x1=5, y1=10, line=dict(color="RoyalBlue", width=3))
    fig.add_shape(type="line", y0=5, x0=0, y1=5, x1=10, line=dict(color="RoyalBlue", width=3))

    # fig.update_layout(title='キーワード単位の評価軸マッピング', xaxis_title=x_axis_custom, yaxis_title=y_axis_custom)
    #st.plotly_chart(fig, use_container_width=True)

    # キーワード単位のベクトル類似度がでているため、それらを表にしてここに表示
    #st.write("キーワード単位の類似度スコア")
    df_similarity = pd.DataFrame({
        'データジャケット名': keywords_data['データジャケット名'],
        'キーワード': keywords_data['キーワード'],
        'X軸類似度': keywords_data['x_similarity'],
        'Y軸類似度': keywords_data['y_similarity']
    })

    # データジャケット単位でベクトルを集約する関数
    def aggregate_vectors(descriptions):
        embeddings = [get_bert_embedding(desc).squeeze() for desc in descriptions]  # .numpy()の呼び出しを削除
        return np.mean(embeddings, axis=0)

    # aggregate_vectors関数の修正により、dj_vectorsの計算
    dj_vectors = keywords_data.groupby('データジャケット名')['キーワードの説明'].apply(aggregate_vectors)

    # 類似度計算関数の入力をチェックし、1-Dであることを確認
    def calculate_similarity(embedding1, embedding2):
        # ここではembedding1, embedding2がnumpy配列であると仮定
        if embedding1.ndim > 1:
            embedding1 = embedding1.squeeze()
        if embedding2.ndim > 1:
            embedding2 = embedding2.squeeze()
        similarity = 1 - cosine(embedding1, embedding2)
        return similarity

    # データジャケット単位で類似度計算
    dj_x_similarity = dj_vectors.apply(lambda x: calculate_similarity(x, x_axis_embedding))
    dj_y_similarity = dj_vectors.apply(lambda x: calculate_similarity(x, y_axis_embedding))

    # 類似度を標準化して0〜10点満点でスケール変換し、グラフの中心を5に設定
    dj_x_similarity_scaled = (dj_x_similarity - dj_x_similarity.min()) / (dj_x_similarity.max() - dj_x_similarity.min()) * 10
    dj_y_similarity_scaled = (dj_y_similarity - dj_y_similarity.min()) / (dj_y_similarity.max() - dj_y_similarity.min()) * 10

    # データジャケット単位のグラフ作成
    fig_dj = go.Figure(data=[go.Scatter(
        x=dj_x_similarity_scaled,
        y=dj_y_similarity_scaled,
        mode='markers+text',
        text=dj_vectors.index,
        marker=dict(size=12, color=dj_x_similarity_scaled, colorscale='Viridis', showscale=True)
    )])

    fig_dj.update_layout(
        title='データジャケット単位の評価軸マッピング')

    if x_axis_selection == "その他（自由記述）" and y_axis_selection == "その他（自由記述）":
        fig_dj.update_layout(
        xaxis_title=x_axis_selection,
        yaxis_title=y_axis_selection)
    elif x_axis_selection != "その他（自由記述）" and y_axis_selection == "その他（自由記述）":
        fig_dj.update_layout(
        xaxis_title=x_axis_selection,
        yaxis_title=y_axis_custom)
    elif x_axis_selection == "その他（自由記述）" and y_axis_selection != "その他（自由記述）":
        fig_dj.update_layout(
        xaxis_title=x_axis_custom,
        yaxis_title=y_axis_selection)
    else:
        fig_dj.update_layout(
        xaxis_title=x_axis_selection,
        yaxis_title=y_axis_selection)
    # ゼロラインを5に設定して、四象限の中心が5で交わるようにする
    fig_dj.add_shape(type="line", x0=5, y0=0, x1=5, y1=10, line=dict(color="RoyalBlue", width=3))
    fig_dj.add_shape(type="line", y0=5, x0=0, y1=5, x1=10, line=dict(color="RoyalBlue", width=3))
    #st.plotly_chart(fig_dj, use_container_width=True)

    # データジャケット単位のベクトル類似度がでているため、それらを表にしてここに表示
    #st.write("データジャケット単位の類似度スコア")
    df_dj_similarity = pd.DataFrame({
        'データジャケット名': dj_vectors.index,
        'X軸類似度': dj_x_similarity_scaled,
        'Y軸類似度': dj_y_similarity_scaled
    })

    # 結果をsession_stateに保存
    save_step5_results(fig, fig_dj,df_similarity,df_dj_similarity)

# ステップ5の結果を表示
display_step5_results()

    # ここでユーザーによるグループ化の入力を受け取る
    #st.markdown("#### データジャケットのグループ化")
    #grouped_dj_input = st.text_area("上記のデータジャケットをグループ化した結果を貼り付けてください", placeholder="例：グループ1: データジャケットA, データジャケットB; グループ2: データジャケットC")

# ステップ6: データジャケットのグループ化結果のアップロード
st.markdown("##### Step6: データジャケットのグループ化結果のアップロード")
st.markdown("データジャケットのグループ化結果が含まれるExcelファイルをアップロードしてください。")

# ファイルアップローダーを使用して、グループ化結果のExcelファイルをアップロードする
uploaded_grouped_file = st.file_uploader("グループ化結果のファイル選択", type=['xlsx'])

# アップロードされたファイルがある場合、データを読み込む
if uploaded_grouped_file is not None:
    df_grouped = pd.read_excel(uploaded_grouped_file)
    st.write("アップロードされたグループ化結果のプレビュー:")
    st.dataframe(df_grouped.head())

    # グループ化されたデータの抽出
    grouped_texts = df_grouped.groupby('クラスタ名')['データジャケット名'].apply(list).to_dict()

    # 各グループに対して、x軸とy軸の評価軸との類似度を計算
    grouped_x_similarities = {}
    grouped_y_similarities = {}
    for group_name, dj_names in grouped_texts.items():
        texts = " ".join(dj_names) # グループ内のデータジャケット名を連結
        group_vector = get_bert_embedding(texts)
        x_axis_vector = get_bert_embedding(x_axis_selection if x_axis_selection != "その他（自由記述）" else x_axis_custom)
        y_axis_vector = get_bert_embedding(y_axis_selection if y_axis_selection != "その他（自由記述）" else y_axis_custom)
        x_similarity = calculate_similarity(group_vector, x_axis_vector)
        y_similarity = calculate_similarity(group_vector, y_axis_vector)
        grouped_x_similarities[group_name] = x_similarity
        grouped_y_similarities[group_name] = y_similarity

    # 類似度を基にグラフ表示
    fig_group = go.Figure(data=[go.Scatter(
        x=list(grouped_x_similarities.values()),
        y=list(grouped_y_similarities.values()),
        mode='markers+text',
        text=list(grouped_x_similarities.keys()),
        marker=dict(size=12, color=np.arange(len(grouped_x_similarities)), colorscale='Viridis', showscale=True)
    )])
    fig_group.update_layout(title='グループと評価軸の類似度マッピング', xaxis_title="x軸の類似度", yaxis_title="y軸の類似度")
    st.plotly_chart(fig_group, use_container_width=True)
