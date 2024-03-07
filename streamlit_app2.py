import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine
import scipy.stats
import numpy as np
from io import BytesIO

# BERTモデルのロード
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# ベクトル化関数を修正して、入力が文字列かどうかをチェック
def get_bert_embedding(text):
    # textが文字列でない場合、エラーメッセージを表示
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, but got {type(text)}")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().squeeze()

# 類似度計算関数の定義
def calculate_similarity(embedding1, embedding2):
    if embedding1.ndim > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.ndim > 1:
        embedding2 = embedding2.squeeze()
    return 1 - cosine(embedding1, embedding2)

# ステップ5の結果を保存する関数
def save_step5_results(fig, fig_dj,df_similarity,df_dj_similarity,df_excel):
    st.session_state['df_similarity'] = df_similarity
    st.session_state['fig'] = fig
    st.session_state['df_dj_similarity'] = df_dj_similarity
    st.session_state['fig_dj'] = fig_dj
    st.session_state['df_excel'] = df_excel

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
    if 'df_excel' in st.session_state:
        st.download_button(label='📥 データジャケットの類似度スコアをダウンロード',
                    data=st.session_state['df_excel'],
                    file_name='dj_similarity_scores.xlsx',
                    mime='application/vnd.ms-excel')

# 既に計算されているdfを使用してExcelファイルを作成し、ダウンロードリンクを提供
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

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
            # データジャケット名列を追加し、ユーザー課題であることを示す
            keywords_df['データジャケット名'] = 'ユーザー課題'
            keywords_df['データジャケット分類'] = 'ユーザー課題'
            # キーワードの説明が空（Noneまたは空文字）の行を削除
            keywords_df.dropna(subset=['キーワードの説明'], inplace=True)
            keywords_df = keywords_df[keywords_df['キーワードの説明'] != '']
            # キーワードの説明がNoneの場合、キーワードを使用
            #keywords_df['キーワードの説明'] = keywords_df['キーワードの説明'].fillna(keywords_df['キーワード'])
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
    st.write(combined_df)

    # 評価軸のベクトル化
    if x_axis_selection == "その他（自由記述）":
        x_axis_embedding = get_bert_embedding(x_axis_custom)
    else:
        x_axis_embedding = get_bert_embedding(x_axis_selection)
    if y_axis_selection == "その他（自由記述）":
        y_axis_embedding = get_bert_embedding(y_axis_custom)
    else:
        y_axis_embedding = get_bert_embedding(y_axis_selection)

    # combined_dfに対してベクトル化を適用する前に、'キーワードの説明'列が文字列型であることを確認
    combined_df['キーワードの説明'] = combined_df['キーワードの説明'].astype(str)
    combined_df['ベクトル'] = combined_df['キーワードの説明'].apply(get_bert_embedding)

    # 評価軸のベクトル化
    x_axis_embedding = get_bert_embedding(x_axis_custom if x_axis_selection == "その他（自由記述）" else x_axis_selection)
    y_axis_embedding = get_bert_embedding(y_axis_custom if y_axis_selection == "その他（自由記述）" else y_axis_selection)

    # 類似度計算
    combined_df['x_similarity'] = combined_df['ベクトル'].apply(lambda x: calculate_similarity(x, x_axis_embedding))
    combined_df['y_similarity'] = combined_df['ベクトル'].apply(lambda x: calculate_similarity(x, y_axis_embedding))

    # スケーリング処理
    combined_df['x_similarity_scaled'] = (combined_df['x_similarity'] - combined_df['x_similarity'].min()) / (combined_df['x_similarity'].max() - combined_df['x_similarity'].min()) * 10
    combined_df['y_similarity_scaled'] = (combined_df['y_similarity'] - combined_df['y_similarity'].min()) / (combined_df['y_similarity'].max() - combined_df['y_similarity'].min()) * 10

    # 新しいグラフの作成
    fig = go.Figure()
    for source, group_df in combined_df.groupby('source'):
        fig.add_trace(go.Scatter(
            x=group_df['x_similarity_scaled'],
            y=group_df['y_similarity_scaled'],
            mode='markers+text',
            text=group_df['キーワード'],
            marker=dict(
                size=12,
                color='red' if source == 'ユーザー課題' else 'blue',  # ユーザー課題は赤、データジャケットは青で表示
            ),
            name=source
        ))

    # グラフ設定
    #fig_new.update_layout(
    #    title='キーワード単位の評価軸マッピング',
    #    xaxis_title='x軸の類似度',
    #    yaxis_title='y軸の類似度',
    #    legend_title='データソース'
    #)

    # 四象限の中心を5で交わるようにゼロラインを追加
    #fig.add_shape(type="line", x0=5, y0=0, x1=5, y1=10, line=dict(color="RoyalBlue", width=3))
    #fig.add_shape(type="line", y0=5, x0=0, y1=5, x1=10, line=dict(color="RoyalBlue", width=3))

    #st.plotly_chart(fig_new, use_container_width=True)

    # キーワードごとに類似度計算
    #keywords_data['x_similarity'] = keywords_data['キーワードの説明'].apply(lambda x: calculate_similarity(get_bert_embedding(x), x_axis_embedding))
    #keywords_data['y_similarity'] = keywords_data['キーワードの説明'].apply(lambda x: calculate_similarity(get_bert_embedding(x), y_axis_embedding))

    # 類似度を標準化して0〜10点満点でスケール変換し、グラフの中心を5に設定
    #keywords_data['x_similarity'] = (keywords_data['x_similarity'] - keywords_data['x_similarity'].min()) / (keywords_data['x_similarity'].max() - keywords_data['x_similarity'].min()) * 10
    #keywords_data['y_similarity'] = (keywords_data['y_similarity'] - keywords_data['y_similarity'].min()) / (keywords_data['y_similarity'].max() - keywords_data['y_similarity'].min()) * 10

    # キーワード単位のグラフ
    #fig = go.Figure(data=[go.Scatter(
    #    x=keywords_data['x_similarity'],
    #    y=keywords_data['y_similarity'],
    #    mode='markers+text',
    #    text=keywords_data['キーワード'],
    #    marker=dict(size=12, color=keywords_data['x_similarity'], colorscale='Viridis', showscale=True)
    #)])

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
        'データジャケット名': combined_df['データジャケット名'],
        'キーワード': combined_df['キーワード'],
        'X軸類似度': combined_df['x_similarity'],
        'Y軸類似度': combined_df['y_similarity']
    })

    # データジャケット単位でベクトルを集約する関数
    def aggregate_vectors(descriptions):
        embeddings = [get_bert_embedding(desc).squeeze() for desc in descriptions]  # .numpy()の呼び出しを削除
        return np.mean(embeddings, axis=0)

    # データジャケット単位でキーワードの類似度スコアを平均する
    dj_avg_similarity = combined_df.groupby('データジャケット名').agg({'x_similarity':'mean', 'y_similarity':'mean'}).reset_index()

    # 類似度のスケーリング
    dj_avg_similarity['x_similarity_scaled'] = (dj_avg_similarity['x_similarity'] - dj_avg_similarity['x_similarity'].min()) / (dj_avg_similarity['x_similarity'].max() - dj_avg_similarity['x_similarity'].min()) * 10
    dj_avg_similarity['y_similarity_scaled'] = (dj_avg_similarity['y_similarity'] - dj_avg_similarity['y_similarity'].min()) / (dj_avg_similarity['y_similarity'].max() - dj_avg_similarity['y_similarity'].min()) * 10


    # データジャケット単位のグラフ作成
    fig_dj = go.Figure()
    for _, row in dj_avg_similarity.iterrows():
        fig_dj.add_trace(go.Scatter(
            x=[row['x_similarity_scaled']],
            y=[row['y_similarity_scaled']],
            mode='markers+text',
            text=[row['データジャケット名']],
            marker=dict(
                size=12,
                color='red' if row['データジャケット名'] == 'ユーザー課題' else 'blue',  # ユーザー課題は赤、それ以外のデータジャケットは青で表示
            ),
            name=row['データジャケット名']
        ))

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
        'データジャケット名': dj_avg_similarity['データジャケット名'],
        'X軸類似度': dj_avg_similarity['x_similarity_scaled'],
        'Y軸類似度': dj_avg_similarity['y_similarity_scaled'],
    })
    df_excel = to_excel(df_dj_similarity)
    #st.download_button(label='📥 データジャケットの類似度スコアをダウンロード',
    #                data=df_excel,
    #                file_name='dj_similarity_scores.xlsx',
    #                mime='application/vnd.ms-excel')

    # 結果をsession_stateに保存
    save_step5_results(fig, fig_dj,df_similarity,df_dj_similarity,df_excel)

# ステップ5の結果を表示
display_step5_results()

    # ここでユーザーによるグループ化の入力を受け取る
    #st.markdown("#### データジャケットのグループ化")
    #grouped_dj_input = st.text_area("上記のデータジャケットをグループ化した結果を貼り付けてください", placeholder="例：グループ1: データジャケットA, データジャケットB; グループ2: データジャケットC")
st.markdown("""
データジャケットのグループ化のためのプロンプトの例

> （データジャケットの類似度スコアをアップロード）データジャケットを類似度も含めて判断してグループ化し、そのグループの名称をデータの一番右に追加して出力してください。グループの名称が入る項目は「クラスタ名」とし、そのグループに適した名前（例：経済指標、社会環境、顧客動向などふさわしい名称を考えてください）としてください。出力はExcelにしてください。ユーザー課題のグループ名はユーザー課題としてください。
***
""")
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

    # クラスタ名で集計しつつ、類似度は平均を取る
    aggregated_data = df_grouped.groupby('クラスタ名').agg({'X軸類似度': 'mean', 'Y軸類似度': 'mean'})

    # 類似度を基にグラフ表示（ユーザー課題は赤色でマッピング、他は青系）
    fig_group = go.Figure()
    for cluster_name, row in aggregated_data.iterrows():
        fig_group.add_trace(go.Scatter(
            x=[row['X軸類似度']],
            y=[row['Y軸類似度']],
            mode='markers+text',
            name=cluster_name,
            text=[cluster_name],
            marker=dict(size=12, color='red' if 'ユーザー課題' in cluster_name else 'blue')
        ))
    fig_group.update_layout(title='グループと評価軸の類似度マッピング', xaxis_title="x軸の類似度", yaxis_title="y軸の類似度")
    st.plotly_chart(fig_group, use_container_width=True)
