import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
from zipfile import ZipFile
from geopandas.tools import sjoin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from shapely.geometry import Point
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from plotly.subplots import make_subplots

# 事前設定
# 日本の中心地点を設定
japan_center = {"lat": 36.2048, "lon": 138.2529}

# 赤系の色を定義
colors_red = ["#FF5733", "#C70039", "#900C3F", "#581845", "#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845", "#FFC300"]


# Streamlitアプリのタイトル設定
st.title('DataUniverse PoC')

# 進捗バーを初期化
progress_bar = st.progress(0)

##############################################################################################################################
################################################# Step1 ######################################################################
##############################################################################################################################
step_text_1 = st.sidebar.markdown('## Step1:ファイルアップロード')
# 現在の作業ディレクトリを取得
current_working_directory = os.getcwd()

# ユーザーがフォルダパスを入力（デフォルト値は現在の作業ディレクトリ）
data_folder = st.sidebar.text_input('フォルダパスを入力してください:', value=current_working_directory)

# デフォルトのファイル名を定義
default_file_names = [
    "地価公示.geojson",
    "将来人口.zip",
    "city_code.csv",
    "商業動態調査（百貨店・スーパー）.csv"
]

# ユーザーがファイル名を入力（デフォルト値は上記で定義したもの）
file_names = [
    st.sidebar.text_input('地価公示データのファイル名を入力してください:', value=default_file_names[0]),
    st.sidebar.text_input('将来人口データのファイル名を入力してください:', value=default_file_names[1]),
    st.sidebar.text_input('CITY_CODEマスタのファイル名を入力してください:', value=default_file_names[2]),
    st.sidebar.text_input('商業動態調査データのファイル名を入力してください:', value=default_file_names[3])
]

# ファイルの存在を確認
missing_files = []
for file_name in file_names:
    file_path = os.path.join(data_folder, file_name)
    if not os.path.exists(file_path):
        missing_files.append(file_name)

if missing_files:
    # 一つでもファイルが見つからない場合はエラーメッセージを表示
    st.sidebar.error("以下のファイルが見つかりません: " + ", ".join(missing_files))
else:
    # すべてのファイルが見つかった場合の処理をここに記述
    st.sidebar.success("すべてのファイルが正常に見つかりました。")
    try:
        # ファイルパスを組み立て
        uploaded_file = os.path.join(data_folder, "地価公示.geojson")
        uploaded_file2 = os.path.join(data_folder, "将来人口.zip")
        uploaded_city_code = os.path.join(data_folder, "city_code.csv")
        uploaded_csv = os.path.join(data_folder, "商業動態調査（百貨店・スーパー）.csv")

    except Exception as e:
        st.error(f'ファイルの読み込みに失敗しました: {e}')

##############################################################################################################################
################################################# Step2 ######################################################################
##############################################################################################################################
step_text_2 = st.sidebar.markdown('## Step2:分析設定')
# 予測年の設定
selected_year = st.sidebar.selectbox(
        "使用する予測年を選択してください",
        options=["2025", "2030", "2035", "2040", "2045", "2050"],
        index=1  # デフォルトは最初の選択肢
    )
# 選択された年に対応する列名
column_name = f"POP{selected_year}"

# 分析手法の設定
analysis_method = st.sidebar.selectbox(
    "分析手法を選択してください:",
    ["線形回帰", "非線形回帰"],
    index=0 # 線形回帰をデフォルトに設定
)
# 非線形回帰が選択された場合、多項式の次数を選択
degree = 2  # デフォルトの次数
#if analysis_method == "非線形回帰":
#    degree = st.sidebar.slider("多項式の次数を選択してください:", min_value=1, max_value=10, value=2)

# 分析粒度の設定
calculation_method='全体で計算'
#calculation_method = st.sidebar.radio(
#    "対象店舗の分析粒度を選択してください:",
#    ('全体で計算', '平均で計算')
#)

# 分析対象の設定
# プリセット店舗情報を定義
laraport_stores_info = {
    "ららぽーとTOKYO-BAY": {"lat": 35.686139, "lon": 139.990115, "sales": 78700, "area": 102000, "stores":460},
    "アルパーク": {"lat": 34.385203, "lon": 132.455293, "sales": 28600, "area": 90000, "stores":160},
    "ららぽーと甲子園": {"lat": 34.717784, "lon": 135.361450, "sales": 20500, "area": 59000, "stores":150},
    "ラゾーナ川崎プラザ": {"lat": 35.530906, "lon": 139.695928, "sales": 95300, "area": 79000, "stores":330},
    "ららぽーと豊洲": {"lat": 35.655646, "lon": 139.791919, "sales": 40400, "area": 62000, "stores":180},
    "ららぽーと柏の葉": {"lat": 35.893956, "lon": 139.951994, "sales": 24000, "area": 50000, "stores":180},
    "ららぽーと横浜": {"lat": 35.466188, "lon": 139.629775, "sales": 46700, "area": 93000, "stores":280},
    "ダイバーシティ東京プラザ": {"lat": 35.625074, "lon": 139.775213, "sales": 28000, "area": 47000, "stores":150},
    "ららぽーと和泉": {"lat": 34.573795, "lon": 135.483403, "sales": 30200, "area": 55000, "stores":220},
    "ららぽーと富士見": {"lat": 35.856873, "lon": 139.549316, "sales": 49000, "area": 80000, "stores":290},
    "ららぽーと海老名": {"lat": 35.454295, "lon": 139.390111, "sales": 38000, "area": 54000, "stores":260},
    "ららぽーとEXPOCITY": {"lat": 34.811889, "lon": 135.531197, "sales": 54000, "area": 88000, "stores":310},
    "ららぽーと湘南平塚": {"lat": 35.335387, "lon": 139.349243, "sales": 31200, "area": 60000, "stores":250}
    #追加可能…
}

# Streamlit UIで店舗を選択
selected_stores = st.sidebar.multiselect("分析対象の店舗を選択してください:", options=list(laraport_stores_info.keys()))

# 選択された店舗の情報をDataFrameに変換
if selected_stores:
    selected_stores_data = []
    for store in selected_stores:
        store_info = laraport_stores_info[store].copy()  # 店舗情報をコピー
        store_info["name"] = store  # 店舗名を追加
        selected_stores_data.append(store_info)

    # DataFrameを作成し、列名を指定
    df_selected_stores = pd.DataFrame(selected_stores_data)
    df_selected_stores = df_selected_stores.rename(columns={
        "name": "店舗名",
        "lat": "緯度",
        "lon": "経度",
        "sales": "店舗売上（百万円）",
        "area": "店舗面積（m2）",
        "stores": "店舗数"
    })
    # 店舗売上を億円単位に変換
    #df_selected_stores["店舗売上（百万円）"] = df_selected_stores["店舗売上（百万円）"] * 100

    # 列の順番を指定
    df_selected_stores = df_selected_stores[["店舗名", "緯度", "経度", "店舗売上（百万円）", "店舗面積（m2）","店舗数"]]

    # Streamlitで表形式で表示
    st.write("選択された分析対象店舗の情報:", df_selected_stores)
else:
    st.write("分析対象店舗が選択されていません。")

# 選択された店舗の情報を変数に格納
stores_data = []
for store_name in selected_stores:
    store_info = laraport_stores_info[store_name]
    # GeoDataFrameに必要な形式でデータを追加
    stores_data.append({
        "name": store_name,
        "lat": store_info["lat"],
        "lon": store_info["lon"],
        "sales": store_info["sales"],
        "area": store_info["area"],
        "stores": store_info["stores"],
        "geometry": Point(store_info["lon"], store_info["lat"])
    })

# GeoDataFrameの作成
if stores_data:
    stores_gdf = gpd.GeoDataFrame(stores_data, geometry='geometry')
    # 地図に選択された店舗の位置情報をプロット
    if not stores_gdf.empty:
        # 選択された店舗の緯度と経度のリストを取得
        latitudes = stores_gdf['lat'].tolist()
        longitudes = stores_gdf['lon'].tolist()
        store_names = stores_gdf['name'].tolist()
        store_sales = stores_gdf['sales'].tolist()
        store_area = stores_gdf['area'].tolist()
        store_stores = stores_gdf['stores'].tolist()

# 2店舗比較を実施する対象を選択
selected_stores2 = st.sidebar.multiselect("２店舗比較する対象の店舗を2つ選択してください:", options=list(laraport_stores_info.keys()))

# 選択された店舗数が2であることを確認
if len(selected_stores2) != 2:
    # 2店舗以外が選択された場合、エラーメッセージを表示
    st.sidebar.error("２店舗のみ選択してください。")
else:
    st.sidebar.success("２店舗が正しく選択されました。")


# データ取り込みの関数
@st.cache_data
def load_and_process_data_1(uploaded_file):
    # 地価公示データの取り込み
    gdf = gpd.read_file(uploaded_file)
    # 必要なデータ変換処理
    gdf['lat'] = gdf.geometry.y
    gdf['lon'] = gdf.geometry.x
    gdf['L01_006'] = pd.to_numeric(gdf['L01_006'], errors='coerce')
    min_price = gdf['L01_006'].min()
    max_price = gdf['L01_006'].max()
    gdf['color_intensity'] = gdf['L01_006'].apply(lambda x: np.log1p(x - min_price + 1))
    df = gdf[['lat', 'lon', 'L01_006', 'color_intensity', 'L01_022', 'L01_023']]
    linear_scaled_size = (df['L01_006'] - min_price) / (max_price - min_price) * 100 + 0.1
    df['size'] = np.log1p(linear_scaled_size)
    df = df.sort_values(by='L01_006')
    return df,gdf

@st.cache_data
def load_and_process_data_2(uploaded_file2,uploaded_city_code):
    #将来人口データとそれに紐づくCITY_CODEデータの取り込み
    with ZipFile(uploaded_file2, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_ref.extractall(temp_dir)
            # シェープファイルの読み込み
            shp_path = next((os.path.join(temp_dir, filename) for filename in os.listdir(temp_dir) if filename.endswith('.shp')), None)
            if shp_path:
                gdf2 = gpd.read_file(shp_path)
                if uploaded_city_code is not None:
                    # CITY_CODEと県名を繋ぐデータを読み込む（列の位置を指定）
                    city_code_to_prefecture = pd.read_csv(uploaded_city_code, usecols=[2, 3], encoding='cp932')
                    # 列名をわかりやすいものに変更
                    city_code_to_prefecture.columns = ['CITY_CODE', '県名']
                    # 確認
                    city_code_to_prefecture.head()
                    # gdf2に県名を結合
                    gdf2 = gdf2.merge(city_code_to_prefecture, left_on='CITY_CODE', right_on='CITY_CODE', how='left')
                    #st.write(gdf2)
                    gdf2 = gdf2[['県名','POP2020', 'POP2025', 'POP2030', 'POP2035', 'POP2040', 'POP2045','POP2050','geometry']]
    return gdf2

@st.cache_data
def load_and_process_data_3(uploaded_csv):
    #商業動態調査データの加工
    df_nationwide = pd.read_csv(uploaded_csv,header=None)
    #st.write(df_nationwide)
    # 1行目と2行目のデータを文字列型に変換して結合
    new_headers = df_nationwide.iloc[0].astype(str) + '_' + df_nationwide.iloc[1].astype(str)
    new_headers = new_headers.map(lambda x: x.replace('nan_', '').replace('__', '_').rstrip('_'))

    # 3行目以降のデータを新しいデータフレームに設定
    df_new = df_nationwide[2:].reset_index(drop=True)

    # 新しい列名をデータフレームに設定
    df_new.columns = new_headers

    # データ前処理
    # 2020年のデータに絞り込み、百貨店のみのデータに絞り込み
    df_2020 = df_new[(df_new['時間軸_db'] == '2020年') & (df_new['表側_項目_db'] == '合計(百貨店＋スーパー)')]
    df_2020 = df_2020[df_2020['地域_db']!= '東京都'] #東京は外れ値としてモデル作成時は外す

    # 合計売上と売場面積を事業所数で割って、1店舗あたりの平均値を計算
    df_2020['事業所数_店'] = pd.to_numeric(df_2020['事業所数_店'], errors='coerce')
    df_2020['合計_百万円'] = pd.to_numeric(df_2020['合計_百万円'], errors='coerce')
    df_2020['売場面積_千平方m'] = pd.to_numeric(df_2020['売場面積_千平方m'], errors='coerce')

    df_2020['平均売上_百万円/店'] = df_2020['合計_百万円'] / df_2020['事業所数_店']
    df_2020['平均売場面積_千平方m/店'] = df_2020['売場面積_千平方m'] / df_2020['事業所数_店']

    # 不要な列やNaNが含まれる行を削除
    #df_2020 = df_2020.dropna()
    return df_2020

#分析方法
# モデル訓練関数
@st.cache_resource
def train_model(X, y, analysis_method, degree=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if analysis_method == "線形回帰":
        model = LinearRegression()
    else:
        # 非線形回帰: 多項式回帰
        model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return X_train, X_test, y_train, y_test, y_pred, model

#グラフ作成
#@st.cache_resource
def scatter_mapbox(df, lat, lon, size, color, hover_name, hover_data, zoom, opacity, color_continuous_scale):
    # 新しいFigureオブジェクトの作成
    fig = go.Figure()

    # Scattermapboxトレースの追加
    fig.add_trace(go.Scattermapbox(
        lat=df[lat],
        lon=df[lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df[size]*10,
            color=df[color],
            opacity=opacity,
            colorscale=color_continuous_scale
        ),
        text=df[hover_name],
        hoverinfo='text',
        hovertext=[str(df[data]) for data in hover_data]
    ))
    return fig

#@st.cache_resource
def add_scatter_mapbox(fig, df, lat, lon, size, color, hover_name, hover_data, zoom, opacity, color_continuous_scale, size_max=20):
    # マーカーサイズの最大値を適用
    marker_sizes = np.minimum(df[size]*10, size_max)

    # Scattermapboxトレースの追加
    fig.add_trace(go.Scattermapbox(
        lat=df[lat],
        lon=df[lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=marker_sizes,
            color=df[color],
            opacity=opacity,
            colorscale=color_continuous_scale,
            showscale=False  # カラースケールを非表示にする
        ),
        text=df[hover_name],
        hoverinfo='text',
        hovertext=[str(df[data]) for data in hover_data],
        showlegend=False  # 凡例にこのトレースを表示しない
    ))
    return fig

#@st.cache_resource
def store_add_scatter_mapbox(fig, latitudes, longitudes, store_names, store_color, size, opacity, hoverinfo):
    # 凡例を表示するためのフラグ
    #show_legend = True
    for lat, lon, name, color in zip(latitudes, longitudes, store_names,store_color):
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers+text',
            marker=go.scattermapbox.Marker(
                size=size,
                color=color,
                opacity=opacity
            ),
            text=[name],
            name=name,  # 凡例名を最初の店舗のみ設定
            hoverinfo=hoverinfo,
            showlegend=True  # 凡例は最初のトレースのみ表示
        ))
        #show_legend = False  # 2回目以降のトレースでは凡例を非表示にする
    return fig

#@st.cache_resource
def Choroplethmapbox(_df, geojson, locations, color, colorscale, zmin, zmax, opacity, line_width, use_log_scale=False):
    # データの対数変換をオプションとして追加
    if use_log_scale:
        # 対数変換されたデータを用いる
        z_values = np.log1p(_df[color].astype(float))
        # ホバーテキストも対数変換前の値を表示
        hover_text = [f"{color}: {_df.loc[loc, color]}" for loc in locations]
    else:
        # 元のデータをそのまま用いる
        z_values = _df[color].astype(float)
        hover_text = [f"{color}: {_df.loc[loc, color]}" for loc in locations]

    # 新しいFigureオブジェクトの作成とChoroplethmapboxトレースの追加
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=z_values,
        colorscale=colorscale,
        zmin=np.log1p(zmin) if use_log_scale else zmin,  # 対数変換を考慮
        zmax=np.log1p(zmax) if use_log_scale else zmax,  # 対数変換を考慮
        marker_opacity=opacity,
        marker_line_width=line_width,
        text=hover_text,  # ホバーテキストを設定
        hoverinfo="text"  # ホバー情報にテキストのみ表示
    ))
    return fig

#@st.cache_resource
def add_Choroplethmapbox(fig,geojson,locations,z,colorscale,zmin,zmax,opacity,line_width):
    # 選択された年に基づく人口データをメッシュとしてプロット
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=locations,
        z=z,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        marker_opacity=opacity,
        marker_line_width=line_width,
    ))
    return fig

#@st.cache_resource
def graph_setting(fig, mapbox_style, zoom, width, height):
    fig.update_layout(
        legend=dict(x=0, y=1),
        showlegend=True,
        mapbox=dict(
            style=mapbox_style,
            center=dict(lat=japan_center['lat'], lon=japan_center['lon']),
            zoom=zoom
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        width=width,
        height=height
    )
    return fig


#@st.cache_resource
def px_scatter(df,x,y,color,hover_data,labels,title):
    fig = px.scatter(
        data_frame=X_with_pref,
        x=x,
        y=y,
        color=color,
        hover_data=hover_data,
        labels=labels,
        title=title
    )
    return fig

#@st.cache_resource
def store_add_px_scatter(fig,x,y,mode,name,marker,hoverinfo,hovertext,text):
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name,
                             marker=marker,
                             hoverinfo=hoverinfo,
                             hovertext=hovertext,
                             text=text)
    )
    return fig

#@st.cache_resource
def px_graph_setting(fig, width, height):
    fig.update_layout(width=width, height=height)
    return fig

# 2店舗比較
def compare_two_stores(stores_data):
    # 選択された2店舗の実売上、予測売上、店舗面積、ロケーションポテンシャルを比較
    fig = make_subplots(rows=2, cols=2, subplot_titles=("売上比較",  "実売上と予測売上の比較","店舗面積比較", "ロケーションポテンシャル比較"))

    # 実売上と予測売上の比較
    fig.add_trace(go.Bar(x=stores_data['店舗名'], y=stores_data['実売上（百万円）'], name='実売上'), row=1, col=1)
    fig.add_trace(go.Bar(x=stores_data['店舗名'], y=stores_data['予測売上（百万円）'], name='予測売上'), row=1, col=1)

    # 売上の比率（実売上/予測売上）
    sales_ratio = stores_data['実売上（百万円）'] / stores_data['予測売上（百万円）']
    fig.add_trace(go.Scatter(x=stores_data['店舗名'], y=sales_ratio, mode='lines+markers', name='売上比率', marker=dict(color='gold')), row=2, col=2)
    # 店舗面積比較
    fig.add_trace(go.Bar(x=stores_data['店舗名'], y=stores_data['店舗面積（m2）'], name='店舗面積'), row=1, col=2)

    # ロケーションポテンシャル比較
    fig.add_trace(go.Bar(x=stores_data['店舗名'], y=stores_data['ロケーションポテンシャル'], name='ロケーションポテンシャル'), row=2, col=1)

    # グラフレイアウトの設定
    fig.update_layout(height=800, width=1000, title_text="2店舗の詳細比較分析", barmode='group')
    fig.update_yaxes(title_text="比率", row=2, col=2)

    # 平均値の線を各グラフに追加
    averages = {'実売上（百万円）': stores_data['実売上（百万円）'].mean(), '予測売上（百万円）': stores_data['予測売上（百万円）'].mean(), '店舗面積（m2）': stores_data['店舗面積（m2）'].mean(), 'ロケーションポテンシャル': stores_data['ロケーションポテンシャル'].mean()}
    for col, avg in averages.items():
        fig.add_hline(y=avg, line_dash="dot", annotation_text=f"平均: {avg}", annotation_position="top right")
    return fig

##############################################################################################################################
################################################# Step3 ######################################################################
##############################################################################################################################
if uploaded_file is not None and uploaded_file2 is not None and uploaded_csv is not None and uploaded_city_code is not None and len(selected_stores2)==2:
    # データ読み込み
    #df,gdf,gdf2,df_2020 = load_and_process_data(uploaded_file, uploaded_file2, uploaded_csv, uploaded_city_code)
    progress_bar.progress(10)  # 進捗を10%に更新
    df,gdf= load_and_process_data_1(uploaded_file)
    progress_bar.progress(20)  # 進捗を10%に更新
    gdf2= load_and_process_data_2(uploaded_file2,uploaded_city_code)
    progress_bar.progress(30)  # 進捗を10%に更新
    df_2020= load_and_process_data_3(uploaded_csv)

step_text_2 = st.markdown('### Step3:地価公示と将来人口の確認')
if uploaded_file is not None and uploaded_file2 is not None and uploaded_csv is not None and uploaded_city_code is not None and len(selected_stores2)==2:
    # 将来人口のメッシュデータの読み込みと表示
    if column_name in gdf2.columns:
        # GeoJSONデータとその他のパラメータを準備
        geojson = gpd.GeoSeries(gdf2['geometry']).__geo_interface__
        locations = gdf2.index.tolist()
        color = column_name  # 実際の色を決定する列名に置き換えてください
        colorscale = "Burg"
        mapbox_style = "carto-positron"
        zmin=gdf2[column_name].min()
        zmax=gdf2[column_name].max()

        # 将来人口のマッピング
        fig = Choroplethmapbox(
        _df=gdf2,
        geojson=geojson,
        locations=locations,
        color=column_name,  # 直接列名を指定
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        opacity=0.3,
        line_width=0,  # 必要に応じて調整
        use_log_scale=True
        )

        progress_bar.progress(40)

        # 地価公示のマッピング
        fig = add_scatter_mapbox(
            fig=fig,
            df=df,
            lat="lat",
            lon="lon",
            size="size",
            color="L01_006",
            hover_name="L01_022",
            hover_data=["L01_006","L01_023"],
            zoom=4,
            opacity=0.5,
            color_continuous_scale="Temps",
            size_max=20
            )
        # 地図に店舗情報をプロット
        fig = store_add_scatter_mapbox(fig,latitudes, longitudes, store_names, colors_red,14,0.7,'text+lat+lon')
        # グラフの調整
        fig = graph_setting(fig=fig, mapbox_style=mapbox_style, zoom=6, width=1000, height=600)
    st.plotly_chart(fig)

    progress_bar.progress(60)  # 進捗を50%に更新

##############################################################################################################################
################################################# Step4 ######################################################################
##############################################################################################################################
    step_text_4 = st.markdown('### Step4:メッシュごとの地価公示平均値を確認')
    # 前処理: gdf（地価公示データ）とgdf2（将来人口のメッシュデータ）の読み込みはすでに完了しているものとします

    # 空間結合: 地価公示のポイントデータをメッシュデータに結合
    gdf_joined = sjoin(gdf, gdf2, how="inner", op="within")

    # 地価公示価格の平均値の計算: 1kmメッシュごと
    gdf_mean_price = gdf_joined.groupby(gdf_joined.index_right)['L01_006'].mean().reset_index()

    # 平均値をメッシュデータに結合
    gdf2_mean_price = gdf2.merge(gdf_mean_price, left_index=True, right_on='index_right')

    # GeoJSONデータとその他のパラメータを準備
    geojson2 = gpd.GeoSeries(gdf2_mean_price['geometry']).__geo_interface__
    locations2 = gdf2_mean_price.index.tolist()
    color2 = 'L01_006'
    colorscale2 = "Viridis"
    mapbox_style2 = "carto-positron"
    zmin2=gdf2_mean_price['L01_006'].min()
    zmax2=gdf2_mean_price['L01_006'].max()

    # 地価公示価格の平均値に基づいて色分けされた地図をプロット
    fig3 = Choroplethmapbox(
    _df=gdf2_mean_price,
    geojson=geojson2,
    locations=locations2,
    color=color2,  # 直接列名を指定
    colorscale=colorscale2,
    zmin=zmin2,
    zmax=zmax2,
    opacity=0.3,
    line_width=0,  # 必要に応じて調整
    use_log_scale=True
    )
    fig3 = store_add_scatter_mapbox(fig3,latitudes, longitudes, store_names, colors_red,14,0.7,'text+lat+lon')
    fig3 = graph_setting(fig=fig3, mapbox_style=mapbox_style, zoom=6, width=1000, height=600)

    # 地図を表示
    st.plotly_chart(fig3)

    progress_bar.progress(70)  # 進捗を70%に更新

##############################################################################################################################
################################################# Step5 ######################################################################
##############################################################################################################################
    step_text_4 = st.markdown('### Step5:ロケーションポテンシャルを設定')
    # ロケーションポテンシャルの計算式を説明する
    st.write('ロケーションポテンシャルは、地価公示価格の平均値と選択された年の人口増加率を基に算出します。')

    # 地価公示データの平均値と将来人口の値から、メッシュごとにロケーションポテンシャルを計算
    gdf2_mean_price['population_growth'] = gdf2_mean_price[column_name].astype(float) / gdf2_mean_price['POP2020'].astype(float)
    gdf2_mean_price['location_potential'] = gdf2_mean_price['L01_006'] * gdf2_mean_price['population_growth']

    # GeoJSONデータとその他のパラメータを準備
    geojson3 = gpd.GeoSeries(gdf2_mean_price['geometry']).__geo_interface__
    locations3 = gdf2_mean_price.index.tolist()
    color3 = 'location_potential'
    colorscale3 = "Viridis"
    mapbox_style3 = "carto-positron"
    zmin3=gdf2_mean_price['location_potential'].min()
    zmax3=gdf2_mean_price['location_potential'].max()

    fig4 = Choroplethmapbox(
    _df=gdf2_mean_price,
    geojson=geojson3,
    locations=locations3,
    color=color3,  # 直接列名を指定
    colorscale=colorscale3,
    zmin=zmin3,
    zmax=zmax3,
    opacity=0.3,
    line_width=0,  # 必要に応じて調整
    use_log_scale=True
    )
    fig4 = store_add_scatter_mapbox(fig4,latitudes, longitudes, store_names, colors_red,14,0.7,'text+lat+lon')
    fig4 = graph_setting(fig=fig4, mapbox_style=mapbox_style, zoom=6, width=1000, height=600)

    st.plotly_chart(fig4)

    progress_bar.progress(80)  # 進捗を90%に更新

##############################################################################################################################
################################################# Step6 ######################################################################
##############################################################################################################################
    st.header('Step6: 売上予測モデルの作成')
    st.markdown('商業動態調査を用いて、１店舗あたりの店舗面積とロケーションポテンシャルから、売上を予測するモデルを作成')

    # gdf2_mean_price（ロケーションポテンシャルを含むGeoDataFrame）を県単位でグループ化し、平均値を計算
    gdf2_mean_price_pre = gdf2_mean_price.groupby(['県名'])['location_potential'].mean().reset_index()
    location_potential_by_prefecture = gdf2_mean_price_pre.groupby('県名')['location_potential'].mean().reset_index()

    # CSVデータにロケーションポテンシャルを結合
    # CSVからのデータ読み込み
    if uploaded_csv is not None:
        # CSVデータにロケーションポテンシャルを結合平均売上_百万円/店
        df_2020_2 = df_2020.merge(location_potential_by_prefecture, right_on='県名',left_on='地域_db', how='right')
        df_2020_3 = df_2020_2[['県名','平均売場面積_千平方m/店', 'location_potential','平均売上_百万円/店']]
        # データのクリーニング
        # 'X'や他の非数値をNaNに置き換えます
        df_2020_3['平均売場面積_千平方m/店'] = pd.to_numeric(df_2020_3['平均売場面積_千平方m/店'], errors='coerce')
        # NaN値の処理
        df_2020_3.dropna(inplace=True)

    # 分析の実施
        # 説明変数(X)として「売場面積」と「ロケーションポテンシャル」を選択
        X = df_2020_3[['平均売場面積_千平方m/店', 'location_potential']]
        # 目的変数(y)として「売上合計」を選択
        y = df_2020_3['平均売上_百万円/店']

        # 分析を実施する関数に投入
        X_train, X_test, y_train, y_test,y_pred, model = train_model(X, y, analysis_method, degree=2)

        # モデルの評価（テストデータ）
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # モデル評価結果を表示
        st.write(f'モデルのMSE: {mse:.2f}, R2スコア: {r2:.2f}')

        # 予測（全データセット）
        y_pred_full = model.predict(X)

        # モデルの評価
        mse_full = mean_squared_error(y, y_pred_full)
        r2_full = r2_score(y, y_pred_full)

        # モデル評価結果を表示
        st.write(f'モデルのMSE: {mse_full:.2f}, R2スコア: {r2_full:.2f}')

        # 県名のリストを作成（ここでは例としてdf_2020_3データフレームの県名列から取得）
        prefectures = df_2020_3['県名'].tolist()

        # 県名を含む新しいDataFrameを作成
        X_with_pref = df_2020_3[['平均売場面積_千平方m/店', 'location_potential', '県名']].copy()

        # 目的変数(y)をこのDataFrameに追加
        X_with_pref['平均売上_百万円/店'] = y

    # 分析結果のプロット
        # 実測値と予測値の散布図
        fig5 = go.Figure()

        # 予測値と実測値のプロット（X軸とY軸を入れ替える）
        fig5.add_trace(go.Scatter(y=y, x=y_pred_full, mode='markers', name='実測値 vs 予測値',
                                    marker=dict(color='LightSkyBlue', size=10, opacity=0.5),
                                    hoverinfo='text',
                                    text=[f"県名: {pref}<br>予測値: {pred:.2f}<br>実測値: {actual:.2f}" for pref, actual, pred in zip(prefectures, y, y_pred_full)]))

        # 選択した店舗の最大実測値を特定
        max_actual_sales = max(store[1] for store in zip(store_names, store_sales))

        # 予測売上の範囲を設定（最大実測値まで）
        predicted_sales_range = np.linspace(min(y_pred_full), max_actual_sales, 100)

        # 理想線の描画
        fig5.add_trace(go.Scatter(x=predicted_sales_range, y=predicted_sales_range, mode='lines', name='理想線',
                                line=dict(color='pink', width=2, dash='dash')))

        # 選択した店舗の予測売上のプロット
        for store in zip(store_names, store_sales, latitudes, longitudes, store_area, store_stores):
            # ロケーションポテンシャルと予測売上を保存するリスト
            nearest_lps = []
            predicted_sales_list = []

            if calculation_method == '全体で計算':
                # 全体で計算する場合
                store_area_thousand_m2 = store[4] / 1000 # 平方メートルから千平方メートルに変換
                store_sales_million_yen = store[1]  # 実売上（百万円）
            elif calculation_method == '平均で計算':
                # 平均で計算する場合、店舗面積と売上を店舗数で割る
                store_area_thousand_m2 = (store[4] / store[5]) / 1000
                store_sales_million_yen = store[1] / store[5]  # 実売上（百万円）を店舗数で割る

            nearest_lp = gdf2_mean_price.loc[gdf2_mean_price.geometry.distance(Point(store[3], store[2])).idxmin(), 'location_potential']
            predicted_sales = model.predict([[store_area_thousand_m2, nearest_lp]])[0]
            fig5.add_trace(go.Scatter(x=[predicted_sales], y=[store_sales_million_yen], mode='markers+text', name=store[0],
                                    marker=dict(size=12, color='red'),
                                    text=[f"{store[0]}"],
                                    textposition="top center"))

        # グラフレイアウトの設定
        fig5.update_layout(title='予測値と実測値の比較',
                            xaxis_title='予測値',
                            yaxis_title='実測値',
                            margin=dict(r=0, t=0, l=0, b=0),
                            width=1000,
                            height=600)

    # 商業動態調査のデータを確認
        # fig6: 売場面積に基づく実測値
        fig6= px_scatter(
            df=X_with_pref,
            x='平均売場面積_千平方m/店',
            y='平均売上_百万円/店',  # 実測値
            color='平均売上_百万円/店',  # 実測値
            hover_data=['県名'],  # ホバー時に県名を表示
            labels={'x': '売場面積', 'y': '実測値', 'color': '実測値'},
            title='売場面積と実測値'
            )
        # fig7: ロケーションポテンシャルに基づく実測値
        fig7 = px_scatter(
            df=X_with_pref,
            x='location_potential',
            y='平均売上_百万円/店',
            color='平均売上_百万円/店',  # 実測値
            hover_data=["県名"],  # ホバー時に県名を表示
            labels={'x': 'ロケーションポテンシャル', 'y': '実測値', 'color': '実測値'},
            title='ロケーションポテンシャルと実測値'
            )

    # 各店舗の予測値を計算

    # ロケーションポテンシャルと予測売上を保存するリスト
    nearest_lps = []
    predicted_sales_list = []

    for store in zip(store_names, store_sales, latitudes, longitudes, store_area,store_stores):

        if calculation_method == '全体で計算':
            # 全体で計算する場合
            store_area_thousand_m2 = store[4] / 1000 # 平方メートルから千平方メートルに変換
            store_sales_million_yen = store[1]  # 実売上（百万円）
        elif calculation_method == '平均で計算':
            # 平均で計算する場合、店舗面積と売上を店舗数で割る
            store_area_thousand_m2 = (store[4] / store[5]) / 1000
            store_sales_million_yen = store[1] / store[5]  # 実売上（百万円）を店舗数で割る

        # 店舗の位置
        store_point = Point(store[3], store[2])

        # 最も近いメッシュのロケーションポテンシャルを検索
        distances = gdf2_mean_price.geometry.distance(store_point)
        nearest_index = distances.idxmin()
        nearest_lp = gdf2_mean_price.loc[nearest_index, 'location_potential']

        nearest_lps.append(nearest_lp)

        # 予測売上の計算
        predicted_sales = model.predict([[store_area_thousand_m2, nearest_lp]])[0]
        predicted_sales_list.append(predicted_sales)

        # 各グラフに店舗情報を追加
        fig6 = store_add_px_scatter(fig6,x=[store_area_thousand_m2], y=[store_sales_million_yen], mode='markers+text', name=store[0],
                                marker=dict(color=colors_red, size=10),
                                hoverinfo='text',
                                hovertext=f"店舗名: {store[0]}<br>実売上: {store_sales_million_yen}百万円<br>店舗面積： {store_area_thousand_m2}千平方m",
                                text=f"{store[0]}")
        fig7 = store_add_px_scatter(fig7,x=[nearest_lp], y=[store_sales_million_yen], mode='markers+text', name=store[0],
                                marker=dict(color=colors_red, size=10),
                                hoverinfo='text',
                                hovertext=f"店舗名: {store[0]}<br>実売上: {store_sales_million_yen}百万円<br>ロケーションポテンシャル：{nearest_lp}",
                                text=f"{store[0]}")

    fig6 = px_graph_setting(fig6, 1000, 600)
    fig7 = px_graph_setting(fig7, 1000, 600)

    # Streamlitでグラフを表示
    st.plotly_chart(fig6)
    st.plotly_chart(fig7)
    st.plotly_chart(fig5)

    # 店舗情報と予測結果を含むDataFrameの作成
    store_evaluation = pd.DataFrame({
        '店舗名': store_names,
        '実売上（百万円）': [store_sales_million_yen for store_sales_million_yen in store_sales],
        '予測売上（百万円）': predicted_sales_list,
        '店舗面積（m2）': store_area,
        'ロケーションポテンシャル':nearest_lps,
        '店舗数': store_stores
    })

    # 全体で計算の場合はそのまま使用、平均で計算の場合は予測売上を店舗数で乗算して全体の予測売上を求める
    if calculation_method == '平均で計算':
        store_evaluation['予測売上（百万円）'] = store_evaluation.apply(lambda x: x['予測売上（百万円）'] * x['店舗数'], axis=1)

    # 売上の妥当性評価
    store_evaluation['評価'] = store_evaluation.apply(lambda row: 'OK' if row['実売上（百万円）'] > row['予測売上（百万円）'] else 'NG', axis=1)

    # 表示用にDataFrameのコピーを作成
    store_evaluation_display = store_evaluation.copy()

    # 評価結果に基づいて色を設定（ここでは例としてテキストで色を指定）
    store_evaluation_display['評価'] = store_evaluation_display['評価'].map({'OK': 'OK', 'NG': 'NG'})

    # 表形式で結果を表示
    st.write(store_evaluation_display)
    progress_bar.progress(90)  # 進捗を90%に更新

##############################################################################################################################
################################################# Step7 ######################################################################
##############################################################################################################################
    st.header('Step7: ２店舗比較による詳細分析')

    progress_bar.progress(100)  # 進捗を100%に更新、処理完了

    # 選択された店舗のデータを取得
    selected_stores_data = store_evaluation_display[store_evaluation_display['店舗名'].isin(selected_stores2)]

    # 選択された店舗のデータを表示
    if not selected_stores_data.empty:
        st.write("選択された店舗のデータ:", selected_stores_data)

        # 2店舗が選択された場合、比較分析を行う
        if len(selected_stores2) == 2:
            fig10 = compare_two_stores(selected_stores_data)
    else:
        st.write("分析対象店舗が選択されていません。")

    # グラフの表示
    st.plotly_chart(fig10)
