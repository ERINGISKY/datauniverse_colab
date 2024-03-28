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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 日本の中心地点を設定
japan_center = {"lat": 36.2048, "lon": 138.2529}

# Streamlitアプリのタイトル設定
st.title('DataUniverse PoC')

# 進捗バーを初期化
progress_bar = st.progress(0)


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

st.write(uploaded_file)
st.write(uploaded_csv)


#step_text_1 = st.sidebar.markdown('## Step1:ファイルアップロード')
#sumple_text_1 = st.sidebar.markdown("#### 地価公示データアップロード")
## 地価公示データの読み込みとプロット
#uploaded_file = st.sidebar.file_uploader("地価公示データのgeojsonファイルをアップロードしてください", type=['geojson'])
#
#sumple_text_2 = st.sidebar.markdown("#### 将来人口データアップロード")
## 将来人口のメッシュデータの読み込みと表示
#uploaded_file2 = st.sidebar.file_uploader("将来人口のメッシュデータ（ZIP形式）をアップロードしてください", type=['zip'])
#uploaded_city_code = st.sidebar.file_uploader("CITY_CODEマスタCSVファイルをアップロードしてください", type=['csv'])
selected_year = st.sidebar.selectbox(
        "使用する予測年を選択してください",
        options=["2025", "2030", "2035", "2040", "2045", "2050"],
        index=1  # デフォルトは最初の選択肢
    )
# CSVデータアップロード
#sumple_text_csv = st.sidebar.markdown("#### 商業動態調査データアップロード")
#uploaded_csv = st.sidebar.file_uploader("商業動態調査データCSVファイルをアップロードしてください", type=['csv'])

sumple_text_3 = st.sidebar.header("分析対象SC情報入力")
sumple_text_4 = st.sidebar.markdown("#### 店舗１")
# 店舗1の情報入力
store1_name = st.sidebar.text_input("名前", "ららぽーとTOKYO-BAY")
store1_lat = st.sidebar.number_input("緯度", value=35.686139)
store1_lon = st.sidebar.number_input("経度", value=139.990115)
store1_sales = st.sidebar.number_input("売上高(百万円)", value=78700)
store1_area = st.sidebar.number_input("商業施設面積(m2)", value=102000)
#store1_stores = st.sidebar.number_input("店舗数(店)", value=440)
#store1_cars = st.sidebar.number_input("駐車台数(台)", value=7000)

sumple_text_5 = st.sidebar.markdown("#### 店舗２")
# 店舗2の情報入力
store2_name = st.sidebar.text_input("店舗2 名前", "ららぽーとEXPOCITY")
store2_lat = st.sidebar.number_input("店舗2 緯度", value=34.806191)
store2_lon = st.sidebar.number_input("店舗2 経度", value=135.534622)
store2_sales = st.sidebar.number_input("売上高(百万円)", value=54000)
store2_area = st.sidebar.number_input("商業施設面積(m2)", value=71000)
#store2_stores = st.sidebar.number_input("店舗数(店)", value=312)
#store2_cars = st.sidebar.number_input("駐車台数(台)", value=4000)

# 店舗の位置をGeoDataFrameに変換
stores_gdf = gpd.GeoDataFrame({'name': [store1_name, store2_name],
                               'sales': [store1_sales, store2_sales],
                               'area': [store1_area, store2_area],
                               'geometry': gpd.points_from_xy([store1_lon, store2_lon], [store1_lat, store2_lat])})

@st.cache_data
def load_and_process_data_1(uploaded_file):
    #progress_bar.progress(10)  # 進捗を10%に更新
    #地価公示データの取り込みと前処理
    #with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
    #    tmp.write(uploaded_file.read())
    #    tmp.seek(0)
    #    gdf = gpd.read_file(tmp.name)
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
    #with tempfile.TemporaryDirectory() as temp_dir:
    #    zip_path = os.path.join(temp_dir, uploaded_file2.name)
    #    with open(zip_path, 'wb') as f:
    #        f.write(uploaded_file2.read())
    #    with ZipFile(zip_path, 'r') as zip_ref:
    #        zip_ref.extractall(temp_dir)
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

    # 新しいデータフレームの先頭を表示して結果を確認
    #st.write(df_new)

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

@st.cache_resource
def train_model2(X, y, degree=2):
    # データを訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 多項式回帰モデルを作成し、訓練
    # make_pipelineを使用して、多項式特徴量の生成と線形回帰モデルの訓練を一連のステップとして実行
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)

    # テストデータセットで予測を実施
    y_pred = model.predict(X_test)

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 評価結果とモデルを返す
    return X_train, X_test, y_train, y_test, y_pred, model

@st.cache_resource
def train_model(X, y):
    # データを訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 線形回帰モデルを作成し、訓練
    model = LinearRegression()
    model.fit(X_train, y_train)

    # テストデータセットで予測を実施
    y_pred = model.predict(X_test)
    return X_train, X_test, y_train, y_test, y_pred, model

    # データを訓練セットとテストセットに分割
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LightGBMのデータセット形式に変換
    #train_data = lgb.Dataset(X_train, label=y_train)
    #test_data = lgb.Dataset(X_test, label=y_test)

    # モデルのパラメータ
    #params = {
    #    'objective': 'fair',  # 'huber'も可能
    #    'metric': 'rmse',
        #'fair_c': 1.0,  # Fair損失関数のパラメータ、必要に応じて調整
    #    'verbosity': -1
    #}

    # モデルを訓練
    #model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000)

    # テストデータで予測
    #y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    #return X_train, X_test, y_train, y_test, y_pred, model

if uploaded_file is not None and uploaded_file2 is not None and uploaded_csv is not None and uploaded_city_code is not None:
    # データ読み込み
    #df,gdf,gdf2,df_2020 = load_and_process_data(uploaded_file, uploaded_file2, uploaded_csv, uploaded_city_code)
    progress_bar.progress(10)  # 進捗を10%に更新
    df,gdf= load_and_process_data_1(uploaded_file)
    progress_bar.progress(20)  # 進捗を10%に更新
    gdf2= load_and_process_data_2(uploaded_file2,uploaded_city_code)
    progress_bar.progress(30)  # 進捗を10%に更新
    df_2020= load_and_process_data_3(uploaded_csv)

step_text_2 = st.markdown('### Step2:地価公示と将来人口の確認')
# 地価公示データの読み込みとプロット
#uploaded_file = st.file_uploader("GMLファイルをアップロードしてください", type=['geojson'])
if uploaded_file is not None and uploaded_file2 is not None and uploaded_csv is not None and uploaded_city_code is not None:

    progress_bar.progress(40)

    # Plotly Expressでマッピング
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        size="size",
        color="L01_006",
        hover_name="L01_022",
        hover_data=["L01_006", "L01_023"],
        center=japan_center,
        zoom=4,
        opacity=0.5,
        color_continuous_scale=px.colors.diverging.Temps,
        mapbox_style="carto-positron",
    )

    # 店舗情報をプロット
    fig.add_trace(go.Scattermapbox(
        lat=[store1_lat, store2_lat],
        lon=[store1_lon, store2_lon],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, color='red', opacity=0.7),
        text=[store1_name, store2_name],
        name="店舗",
        hoverinfo='text+lat+lon'
    ))

    # Streamlitに地図を表示
    #st.plotly_chart(fig)

    progress_bar.progress(50)  # 進捗を30%に更新

# 将来人口のメッシュデータの読み込みと表示
#uploaded_file2 = st.file_uploader("将来人口のメッシュデータ（ZIP形式）をアップロードしてください", type=['zip'])
if uploaded_file is not None and uploaded_file2 is not None and uploaded_csv is not None and uploaded_city_code is not None:

    # 選択された年に対応する列名
    column_name = f"POP{selected_year}"

    if column_name in gdf2.columns:
        # GeoJSON形式で地理データを取得
        geojson = gpd.GeoSeries(gdf2['geometry']).__geo_interface__

        # 選択された年に基づく人口データをメッシュとしてプロット
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=gdf2.index,
            z=gdf2[column_name].astype(float),  # 人口データ
            colorscale="Burg",
            zmin=gdf2[column_name].min(),
            zmax=gdf2[column_name].max(),
            marker_opacity=0.3,
            marker_line_width=0,
        ))

        # 地図のスタイルと中心地の調整
        fig.update_layout(mapbox_style="carto-positron", mapbox_center=japan_center,mapbox_zoom=6)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                            width=1000,
                            height=600)
        st.plotly_chart(fig)

    progress_bar.progress(60)  # 進捗を50%に更新

#Step3
    step_text_3 = st.markdown('### Step3:メッシュごとの地価公示平均値を確認')
    # 前処理: gdf（地価公示データ）とgdf2（将来人口のメッシュデータ）の読み込みはすでに完了しているものとします

    # 空間結合: 地価公示のポイントデータをメッシュデータに結合
    gdf_joined = sjoin(gdf, gdf2, how="inner", op="within")

    # 地価公示価格の平均値の計算: 1kmメッシュごと
    gdf_mean_price = gdf_joined.groupby(gdf_joined.index_right)['L01_006'].mean().reset_index()
    # 地価公示価格の平均値の計算: 1kmメッシュごと（県名単位）
    # gdf_mean_price_pre = gdf_mean_price.groupby(['県名'])['L01_006'].mean().reset_index()

    # 平均値をメッシュデータに結合
    gdf2_mean_price = gdf2.merge(gdf_mean_price, left_index=True, right_on='index_right')

    # 地図上に表示: gdf2_mean_priceには、各メッシュに対する地価公示価格の平均値が含まれています
    # GeoDataFrameのgeometry列をGeoJSON形式に変換
    geojson2 = gpd.GeoSeries(gdf2_mean_price['geometry']).__geo_interface__

    # 地図の中心を計算
    centroid = gdf2_mean_price.geometry.centroid
    map_center = {"lat": centroid.y.mean(), "lon": centroid.x.mean()}

    # 地価公示価格の平均値に基づいて色分けされた地図をプロット
    fig3 = px.choropleth_mapbox(gdf2_mean_price,
                            geojson=geojson2,
                            locations=gdf2_mean_price.index,
                            color='L01_006',  # ここに平均地価を表す列名を指定
                            color_continuous_scale="Viridis",
                            mapbox_style="carto-positron",
                            #center=map_center,
                            #zoom=-10,
                            opacity=0.3)

    # 地図のスタイルと中心地の調整
    fig3.update_layout(mapbox_style="carto-positron", mapbox_center=japan_center,mapbox_zoom=6)
    fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    width=1000,
                    height=600)

    # 店舗情報をプロット
    fig3.add_trace(go.Scattermapbox(
        lat=[store1_lat, store2_lat],
        lon=[store1_lon, store2_lon],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, color='red', opacity=0.7),
        text=[store1_name, store2_name],
        name="店舗",
        hoverinfo='text+lat+lon'
    ))

    # 地図を表示
    st.plotly_chart(fig3)

    progress_bar.progress(70)  # 進捗を70%に更新

#Step4
    step_text_4 = st.markdown('### Step4:ロケーションポテンシャルを設定')
    # ロケーションポテンシャルの計算式を説明する
    st.write('ロケーションポテンシャルは、地価公示価格の平均値と選択された年の人口増加率を基に算出します。')

    # 地価公示データの平均値と将来人口の値から、メッシュごとにロケーションポテンシャルを計算
    gdf2_mean_price['population_growth'] = gdf2_mean_price[column_name].astype(float) / gdf2_mean_price['POP2020'].astype(float)
    gdf2_mean_price['location_potential'] = gdf2_mean_price['L01_006'] * gdf2_mean_price['population_growth']
    gdf2_mean_price_pre = gdf2_mean_price.groupby(['県名'])['location_potential'].mean().reset_index()
    #st.write(gdf2_mean_price)
    #st.write(gdf2_mean_price_pre)
    # ロケーションポテンシャルに基づいて地図をプロット
    geojson3 = gpd.GeoSeries(gdf2_mean_price['geometry']).__geo_interface__
    fig4 = px.choropleth_mapbox(gdf2_mean_price,
                                geojson=geojson3,
                                locations=gdf2_mean_price.index,
                                color='location_potential',
                                color_continuous_scale=px.colors.sequential.Tealgrn,
                                mapbox_style="carto-positron",
                                center={"lat": gdf2_mean_price.geometry.centroid.y.mean(), "lon": gdf2_mean_price.geometry.centroid.x.mean()},
                                opacity=0.3)
    fig4.update_layout(margin={"r":0, "t":0, "l":0, "b":0}, width=1000, height=600, mapbox_center=japan_center,mapbox_zoom=6)

    # 店舗情報をプロット
    fig4.add_trace(go.Scattermapbox(
        lat=[store1_lat, store2_lat],
        lon=[store1_lon, store2_lon],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=14, color='red', opacity=0.7),
        text=[store1_name, store2_name],
        name="店舗",
        hoverinfo='text+lat+lon'
    ))

    st.plotly_chart(fig4)

    progress_bar.progress(80)  # 進捗を90%に更新

# Step5: 売上の妥当性判断
    st.header('Step5: 入力された店舗の売上が妥当かどうかの判断')

    # gdf2_mean_price（ロケーションポテンシャルを含むGeoDataFrame）を県単位でグループ化し、平均値を計算
    location_potential_by_prefecture = gdf2_mean_price_pre.groupby('県名')['location_potential'].mean().reset_index()

    # CSVデータにロケーションポテンシャルを結合
    # CSVからのデータ読み込み
    if uploaded_csv is not None:
        #st.write("df_2020")
        #st.write(df_2020)
        # CSVデータにロケーションポテンシャルを結合平均売上_百万円/店
        df_2020_2 = df_2020.merge(location_potential_by_prefecture, right_on='県名',left_on='地域_db', how='right')
        #st.write("2020_2")
        #st.write(df_2020_2)
        df_2020_3 = df_2020_2[['県名','平均売場面積_千平方m/店', 'location_potential','平均売上_百万円/店']]
        # データのクリーニング
        # 'X'や他の非数値をNaNに置き換えます
        df_2020_3['平均売場面積_千平方m/店'] = pd.to_numeric(df_2020_3['平均売場面積_千平方m/店'], errors='coerce')
        # NaN値の処理
        # NaN値を0や他の適切な値で置き換えることができます（ここでは0を使用）
        #df_2020_2.fillna(0, inplace=True)
        df_2020_3.dropna(inplace=True)
        #確認用
        #st.write(location_potential_by_prefecture)
        #st.write(df_2020_3)

    # 分析の実施
        # 説明変数(X)として「売場面積」と「ロケーションポテンシャル」を選択
        X = df_2020_3[['平均売場面積_千平方m/店', 'location_potential']]
        # 目的変数(y)として「売上合計」を選択
        y = df_2020_3['平均売上_百万円/店']

        # データを訓練セットとテストセットに分割
        X_train, X_test, y_train, y_test,y_pred, model = train_model2(X, y)

        # モデルの評価
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

        # fig5: 全データセットでの実測値と予測値の比較
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=y, y=y_pred_full, mode='markers', name='実測値 vs 予測値',
                                marker=dict(color='LightSkyBlue', size=10, opacity=0.5),
                                hoverinfo='text',
                                text=["県名: {}<br>実測値: {:.2f}<br>予測値: {:.2f}".format(pref, actual, pred) for pref, actual, pred in zip(prefectures, y, y_pred_full)]))
        fig5.add_trace(go.Scatter(x=y, y=y, mode='lines', name='理想線',
                                line=dict(color='FireBrick', width=2, dash='dash')))
        fig5.update_layout(title='全データセットの実測値と予測値の比較',
                        xaxis_title='実測値',
                        yaxis_title='予測値')
        fig5.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                            width=1000,
                            height=600)
        # Streamlitでグラフを表示
        st.plotly_chart(fig5)

        # fig6: 売場面積に基づく実測値と予測値の比較
        fig6 = px.scatter(
            data_frame=X_with_pref,  # 新しいDataFrameを使用
            x='平均売場面積_千平方m/店',
            y=y_pred_full,  # 予測値
            color='平均売上_百万円/店',  # 実測値
            hover_data=['県名'],  # ホバー時に県名を表示
            labels={'x': '売場面積', 'y': '予測値', 'color': '実測値'},
            title='売場面積に基づく実測値と予測値の比較'
        )
        st.plotly_chart(fig6)

        # fig7: ロケーションポテンシャルに基づく実測値と予測値の比較
        fig7 = px.scatter(
            data_frame=X_with_pref,
            x='location_potential',
            y=y_pred_full,
            color='平均売上_百万円/店',  # 実測値
            hover_data=["県名"],  # ホバー時に県名を表示
            labels={'x': 'ロケーションポテンシャル', 'y': '予測値', 'color': '実測値'},
            title='ロケーションポテンシャルに基づく実測値と予測値の比較')

        # Streamlitでグラフを表示
        st.plotly_chart(fig7)

        progress_bar.progress(90)  # 進捗を90%に更新

    # 予測結果の表示と評価
    for store in [(store1_name, store1_sales, store1_lat, store1_lon, store1_area), (store2_name, store2_sales, store2_lat, store2_lon, store2_area)]:
        # 店舗の位置
        store_point = Point(store[3], store[2])

        # 最も近いメッシュのロケーションポテンシャルを検索
        distances = gdf2_mean_price.geometry.distance(store_point)
        nearest_index = distances.idxmin()
        nearest_lp = gdf2_mean_price.loc[nearest_index, 'location_potential']

        # 予測売上の計算
        predicted_sales = model.predict([[store[4]/1000, nearest_lp]])[0]

        # 実売上と予測売上の比較
        feedback = "がんばっています！" if store[1] > predicted_sales else "もう少し頑張りましょう。"

        # 結果の表示
        st.write(f"{store[0]}の実売上: {store[1]}百万円, 予測売上: {predicted_sales:.2f}百万円。{feedback}")

        progress_bar.progress(100)  # 進捗を100%に更新、処理完了


#同じフォルダにデフォルトアップロードファイルを用意
#いくつかららぽーとのパターンを用意
