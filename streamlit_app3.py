import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go  # Graph Objectsをインポート
import tempfile
import os
from zipfile import ZipFile


# ららぽーとの地点情報
locations = {
    "ららぽーとTOKYO-BAY": (35.686139, 139.990115),
    "ららぽーとEXPOCITY": (34.806191, 135.534622)
}
# st.write(pd.Dataframe(locations))

# Streamlitアプリでファイルアップローダーを提供
uploaded_file = st.file_uploader("GMLファイルをアップロードしてください")
if uploaded_file is not None:
    # 一時ファイルとして保存
    with open("temp_file.gml", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Geopandasで読み込み
    gdf = gpd.read_file("temp_file.gml")

    # 緯度と経度の情報を抽出
    gdf['lat'] = gdf.geometry.y
    gdf['lon'] = gdf.geometry.x

    # 'L01_006'列が文字列型の場合、数値型に変換
    gdf['L01_006'] = pd.to_numeric(gdf['L01_006'], errors='coerce')

    # 地価公示価格に基づく色の調整（逆指数関数的なスケーリング）
    # 地価公示価格の最小値と最大値
    min_price = gdf['L01_006'].min()
    max_price = gdf['L01_006'].max()

    # 逆指数関数的スケーリングの適用
    # 色の調整のためには0より大きい値が必要なので、最小値を引いて正規化し、1を加える
    gdf['color_intensity'] = gdf['L01_006'].apply(lambda x: np.log1p(x - min_price + 1))

    # GeoDataFrameからpandas DataFrameへ変換（必要な列のみを保持）
    df = gdf[['lat', 'lon', 'L01_006', 'color_intensity','L01_022','L01_023']]

    # 地価公示価格の最小値と最大値に基づく線形変換
    linear_scaled_size = (df['L01_006'] - min_price) / (max_price - min_price) * 100 + 0.1  # +10は最小サイズのオフセット

    # 線形変換後の値に対して対数変換を適用
    df['size'] = np.log1p(linear_scaled_size)

    # 地価公示価格で昇順にソート
    df = df.sort_values(by='L01_006')

    # StreamlitにDataFrameを表示
    st.write(df.head())

    # Plotly Expressでマッピング
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        size="size",  # 地価公示価格をサイズで表現、対数スケールを使用
        color="L01_006",  # 地価公示価格を色で表現
        hover_name="L01_022",  # ホバー時に表示する名前
        hover_data=["L01_006","L01_023"],  # ホバー時に追加で表示するデータ
        center={"lat": df["lat"].mean(), "lon": df["lon"].mean()},  # 地図の中心
        opacity=0.3,
        width=800,
        height=800,
        # zoom=10,
        #color_continuous_scale=px.colors.sequential.Plasma,  # 色相のカスタマイズ
        color_continuous_scale=px.colors.diverging.Temps,  # 色相のカスタマイズ
        mapbox_style="carto-positron",
    )

    # ららぽーとの地点を赤色でプロット
    for name, (lat, lon) in locations.items():
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=14,
                color='red',
                opacity=0.7
            ),
            text=[name],
            hoverinfo='text'
        ))
    # Streamlitに地図を表示
    st.plotly_chart(fig)

# Streamlitアプリでファイルアップローダーを提供
uploaded_file2 = st.file_uploader("将来人口のメッシュデータ（ZIP形式）をアップロードしてください", type=['zip'])

if uploaded_file2 is not None:
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        # アップロードされたZIPファイルを一時ディレクトリに保存
        zip_path = os.path.join(temp_dir, uploaded_file2.name)
        with open(zip_path, 'wb') as f:
            f.write(uploaded_file2.getbuffer())

        # ZIPファイルを解凍
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # シェープファイルの読み込み
        shp_path = [os.path.join(temp_dir, filename) for filename in os.listdir(temp_dir) if filename.endswith('.shp')][0]
        gdf2 = gpd.read_file(shp_path)

    # 年を選択するUI（決め打ち）
    selected_year = st.selectbox(
        "年を選択してください",
        options=["2025", "2030", "2035", "2040", "2045", "2050"],
        index=0  # デフォルトは最初の選択肢
    )

    # 選択された年に対応する列名
    column_name = f"POP{selected_year}"

    if column_name in gdf2.columns:
        # GeoJSON形式で地理データを取得
        geojson = gpd.GeoSeries(gdf2['geometry']).__geo_interface__

        fig2 = px.choropleth_mapbox(
            gdf2,
            geojson=geojson,
            locations=gdf2.index,
            color=column_name,  # 正しい列名を使用
            color_continuous_scale=px.colors.sequential.Plasma,
            mapbox_style="carto-positron",
            center={"lat": gdf2.geometry.centroid.y.mean(), "lon": gdf2.geometry.centroid.x.mean()},
            opacity=0.5,
            zoom=5
        )
        fig2.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        st.plotly_chart(fig2)
    else:
        st.error(f"選択された{selected_year}年のデータが見つかりません。")
