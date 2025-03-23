import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
import logging
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_json("data/sales.json")
    except Exception as e:
        st.error("Gagal memuat data. Pastikan file 'data/sales.json' tersedia.")
        logging.error("Error load_data: %s", e)
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    if "avg_price_per_month" not in df.columns or "avg_sales_per_month" not in df.columns:
        df["avg_price_per_month"] = df.groupby(["product_name", "month"])["base_price"].transform("mean")
        df["avg_sales_per_month"] = df.groupby(["product_name", "month"])["sales_volume"].transform("mean")
    logging.info("Data berhasil dimuat dan dipreproses.")
    return df
df = load_data()

if df.empty:
    st.stop()

@st.cache_data(show_spinner=False)
def prepare_encoders(data):
    try:
        le_product = LabelEncoder()
        le_category = LabelEncoder()
        le_customer = LabelEncoder()
        le_channel = LabelEncoder()
        data["product_encoded"] = le_product.fit_transform(data["product_name"])
        data["category_encoded"] = le_category.fit_transform(data["category"])
        data["customer_encoded"] = le_customer.fit_transform(data["customer_segment"])
        data["channel_encoded"] = le_channel.fit_transform(data["channel"])
        scaler = StandardScaler()
        data["base_price_scaled"] = scaler.fit_transform(data[["base_price"]])
    except Exception as e:
        st.error("Error saat melakukan encoding & scaling.")
        logging.error("Error prepare_encoders: %s", e)
        raise e
    return data, le_product, le_category, le_customer, le_channel, scaler
df, le_product, le_category, le_customer, le_channel, scaler = prepare_encoders(df)

features = [
    "month", "year", "product_encoded", "category_encoded",
    "customer_encoded", "channel_encoded", "avg_price_per_month", "avg_sales_per_month"
]

target = "base_price"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train, model_type="RandomForest"):
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model %s berhasil dilatih.", model_type)
    return model
model_rf = train_model(X_train, y_train, model_type="RandomForest")
y_pred = model_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

def rekomendasi_harga(product_name, month, year, category, customer_segment, channel, region="Default"):
    if product_name not in le_product.classes_:
        st.error(f"Produk '{product_name}' tidak terdaftar dalam encoder.")
        return None, None, None, None
    if category not in le_category.classes_:
        st.error(f"Kategori '{category}' tidak terdaftar dalam encoder.")
        return None, None, None, None
    try:
        product_val = le_product.transform([product_name])[0]
        category_val = le_category.transform([category])[0]
        customer_val = le_customer.transform([customer_segment])[0]
        channel_val = le_channel.transform([channel])[0]
    except Exception as e:
        st.error("Error dalam transformasi label. Periksa input produk atau kategori.")
        logging.error("Error rekomendasi_harga label transform: %s", e)
        return None, None, None, None
    data_input = pd.DataFrame({
        "month": [month],
        "year": [year],
        "product_encoded": [product_val],
        "category_encoded": [category_val],
        "customer_encoded": [customer_val],
        "channel_encoded": [channel_val],
        "avg_price_per_month": [df[df["product_name"] == product_name]["base_price"].mean()],
        "avg_sales_per_month": [df[df["product_name"] == product_name]["sales_volume"].mean()]
    })
    pred_price = model_rf.predict(data_input)[0]
    avg_price = df[df["product_name"] == product_name]["base_price"].mean()
    avg_sales = df[df["product_name"] == product_name]["sales_volume"].mean()
    if pred_price > avg_price and avg_sales < 50:
        strategi = "Turunkan harga sedikit untuk meningkatkan penjualan."
        final_price = pred_price * 0.95
    elif pred_price < avg_price and avg_sales > 50:
        strategi = "Naikkan harga karena permintaan tinggi."
        final_price = pred_price * 1.10
    else:
        strategi = "Harga sudah kompetitif, pertahankan strategi saat ini."
        final_price = pred_price
    region_adjustment = {"Default": 1.0, "Urban": 1.05, "Rural": 0.95}
    final_price *= region_adjustment.get(region, 1.0)
    penjelasan = f"""
        - Harga Prediksi untuk {product_name}: **Rp{round(pred_price, 2):,.2f}**
        - Rata-rata Harga Bulanan: **Rp{round(avg_price, 2):,.2f}**
        - Rata-rata Volume Penjualan Bulanan: **{round(avg_sales, 2)} unit**
        - Strategi Penyesuaian Harga: **{strategi}**
        - Harga Final yang Direkomendasikan untuk {region}: **Rp{round(final_price, 2):,.2f}**
    """
    return round(pred_price, 2), strategi, round(final_price, 2), penjelasan

def simulasi_profit_roi(final_harga, cost, volume):
    revenue = final_harga * volume
    profit = revenue - (cost * volume)
    roi = (profit / (cost * volume)) * 100
    return profit, roi

def analisis_kompetitor(product_name):
    product_data = df[df["product_name"] == product_name]
    comp_price = product_data["base_price"].mean()
    return comp_price

def prediksi_harga_saing(product_name, month, year, category, customer_segment, channel, region="Default"):
    if product_name not in le_product.classes_:
        st.error(f"Produk '{product_name}' tidak terdaftar dalam encoder.")
        return None, None
    if category not in le_category.classes_:
        st.error(f"Kategori '{category}' tidak terdaftar dalam encoder.")
        return None, None
    try:
        product_val = le_product.transform([product_name])[0]
        category_val = le_category.transform([category])[0]
        customer_val = le_customer.transform([customer_segment])[0]
        channel_val = le_channel.transform([channel])[0]
    except Exception as e:
        st.error("Error dalam transformasi label. Periksa input produk atau kategori.")
        logging.error("Error prediksi_harga_saing label transform: %s", e)
        return None, None
    data_input = pd.DataFrame({
        "month": [month],
        "year": [year],
        "product_encoded": [product_val],
        "category_encoded": [category_val],
        "customer_encoded": [customer_val],
        "channel_encoded": [channel_val],
        "avg_price_per_month": [df[df["product_name"] == product_name]["base_price"].mean()],
        "avg_sales_per_month": [df[df["product_name"] == product_name]["sales_volume"].mean()]
    })
    pred_price = model_rf.predict(data_input)[0]
    return round(pred_price, 2)

def plot_comparison(product_name, pred_price, comp_price):
    product_data = df[df["product_name"] == product_name]
    harga_produk = product_data["base_price"].mean()
    comparison_data = {
        'Harga Prediksi Anda': pred_price,
        'Harga Rata-rata Kompetitor': comp_price
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(comparison_data.keys()), y=list(comparison_data.values()), palette="Blues_d", ax=ax)
    ax.set_title(f"Perbandingan Harga {product_name}")
    ax.set_ylabel("Harga (Rp)")
    ax.set_ylim(0, max(pred_price, comp_price) * 1.2)
    st.pyplot(fig)
    
def forecast_penjualan(periods, product_name):
    product_data = df[df["product_name"] == product_name]
    if product_data.empty:
        st.error(f"Tidak ada data untuk produk '{product_name}'.")
        return None, None
    product_data['date'] = pd.to_datetime(product_data['date'])
    product_monthly_sales = product_data.groupby(product_data['date'].dt.to_period("M"))['sales_volume'].sum()
    if len(product_monthly_sales) < 2:
        st.error(f"Data untuk produk '{product_name}' tidak cukup untuk forecasting (minimal 2 data point).")
        return None, None
    try:
        model = ExponentialSmoothing(product_monthly_sales, trend='add', seasonal=None, damped_trend=False)
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=periods)
        last_date = product_monthly_sales.index[-1].to_timestamp()
        forecast_index = pd.date_range(start=last_date, periods=periods + 1, freq='M')[1:]
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Sales': forecast_values})
        plt.figure(figsize=(10, 6))
        plt.plot(product_monthly_sales.index.to_timestamp(), product_monthly_sales, label='Aktual', color='blue')
        plt.plot(forecast_df['Date'], forecast_df['Forecasted Sales'], label='Forecast', color='red', linestyle='--')
        plt.title(f"Proyeksi Penjualan {product_name}")
        plt.xlabel("Tanggal")
        plt.ylabel("Volume Penjualan")
        plt.legend()
        st.pyplot(plt)
        rmse = np.sqrt(mean_squared_error(product_monthly_sales.values, model_fit.fittedvalues))
        return forecast_df, rmse
    except Exception as e:
        st.error(f"Kesalahan dalam pemodelan: {str(e)}")
        return None, None
        
def segmentasi_pasar(n_clusters=3, features=None):
    """
    Perform market segmentation using KMeans clustering on product data.
    :param n_clusters: The number of clusters to form (default is 3)
    :param features: The features to use for clustering (default is avg_sales_per_month and avg_price_per_month)
    :return: DataFrame with product names and cluster labels
    """
    if features is None:
        features = ["avg_sales_per_month", "avg_price_per_month"]
    try:
        features_cluster = df.groupby("product_name").agg({
            "avg_sales_per_month": "mean",
            "avg_price_per_month": "mean"
        }).reset_index()
        if len(features_cluster) < n_clusters:
            st.warning(f"Not enough products to form {n_clusters} clusters. Reducing the number of clusters to {len(features_cluster)}.")
            n_clusters = len(features_cluster)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_cluster[features])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        features_cluster["cluster"] = kmeans.fit_predict(scaled_features)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Clustering: Price vs Sales**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x=features_cluster["avg_price_per_month"], 
                y=features_cluster["avg_sales_per_month"], 
                hue=features_cluster["cluster"], 
                palette="viridis", 
                s=100, alpha=0.6, edgecolor='k',
                ax=ax
            )
            ax.set_title("Product Clustering (Sales vs Price)")
            ax.set_xlabel("Average Price per Month")
            ax.set_ylabel("Average Sales per Month")
            st.pyplot(fig)
        with col2:
            st.write(f"**Number of Products per Cluster**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(
                x=features_cluster["cluster"], 
                palette="viridis",
                ax=ax2
            )
            ax2.set_title("Distribution of Products in Clusters")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Number of Products")
            st.pyplot(fig2)
        return features_cluster
    except KeyError as e:
        st.error(f"Missing required column in the dataframe: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error in segmentation: {e}")
        return pd.DataFrame()

def plot_elbow_method():
    """
    Plot the Elbow Method to determine the optimal number of clusters.
    """
    try:
        features_cluster = df.groupby("product_name").agg({
            "avg_sales_per_month": "mean",
            "avg_price_per_month": "mean"
        }).reset_index()
        if len(features_cluster) < 2:
            st.warning("Not enough data to perform the elbow method.")
            return
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_cluster[["avg_sales_per_month", "avg_price_per_month"]])
        inertia = []
        for k in range(1, min(11, len(features_cluster) + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_features)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(inertia) + 1), inertia, marker='o')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in elbow method: {e}")

def deteksi_anomali(product_name, threshold=2):
    """
    Detect anomalies in product prices based on Z-scores.
    :param product_name: The name of the product to analyze.
    :param threshold: The Z-score threshold for detecting anomalies (default is 2).
    :return: A DataFrame containing the detected anomalies.
    """
    product_prices = df[df["product_name"] == product_name]["base_price"]
    if product_prices.isnull().any():
        st.warning(f"Missing values detected for product '{product_name}'.")
        product_prices = product_prices.dropna()
    if len(product_prices) < 2:
        st.warning(f"Not enough data for product '{product_name}' to perform anomaly detection.")
        return pd.DataFrame()
    z_scores = (product_prices - product_prices.mean()) / product_prices.std()
    anomalies = product_prices[np.abs(z_scores) > threshold]
    if anomalies.empty:
        st.info(f"No anomalies detected for product '{product_name}' with the current threshold.")
        return pd.DataFrame()
    anomaly_percentage = (len(anomalies) / len(product_prices)) * 100
    st.subheader(f"Anomaly Detection for Product: {product_name}")
    st.write(f"Detected {len(anomalies)} anomalies ({anomaly_percentage:.2f}%) out of {len(product_prices)} data points.")
    st.write(f"Descriptive Statistics of Anomalies:")
    st.write(anomalies.describe())
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(product_prices)), y=product_prices, label="Product Prices", color='blue')
    sns.scatterplot(x=anomalies.index, y=anomalies, color='red', label="Anomalies", s=100)
    plt.title(f"Anomaly Detection for {product_name}")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)
    anomalies_df = pd.DataFrame({"price": anomalies})
    return anomalies_df

def rekomendasi_stok(product_name):
    """
    Provides stock recommendations based on average sales volume and sales trend.
    :param product_name: The name of the product to analyze.
    :return: Stock recommendation and relevant details.
    """
    product_data = df[df["product_name"] == product_name]
    avg_sales = product_data["sales_volume"].mean()
    sales_trend = product_data.groupby('date')['sales_volume'].sum().pct_change().fillna(0).mean()
    if avg_sales > 100:
        recommendation = "Tingkatkan stok 20% lebih banyak."
        trend_message = "Tren penjualan menunjukkan permintaan yang stabil dan tinggi."
    elif avg_sales < 50:
        recommendation = "Pertahankan stok atau kurangi sedikit."
        trend_message = "Tren penjualan menunjukkan penurunan permintaan."
    else:
        recommendation = "Stok normal."
        trend_message = "Tren penjualan menunjukkan permintaan yang stabil."
    if sales_trend < 0:
        trend_message += " Penurunan penjualan menunjukkan permintaan yang menurun."
    elif sales_trend > 0:
        trend_message += " Kenaikan penjualan menunjukkan permintaan yang meningkat."
    st.subheader("Tren Penjualan Produk")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='sales_volume', data=product_data, marker='o', color='blue', label="Penjualan Bulanan")
    plt.title(f"Tren Penjualan untuk Produk: {product_name}")
    plt.xlabel("Tanggal")
    plt.ylabel("Volume Penjualan")
    st.pyplot(plt)
    return recommendation, trend_message

def penyesuaian_harga_dinamis(current_price, real_time_factor=1.0, demand_factor=1.0, competition_factor=1.0):
    """
    Adjust product price dynamically based on real-time factors, demand, and competition.
    :param current_price: The current base price of the product.
    :param real_time_factor: Factor for real-time pricing adjustment (default is 1.0).
    :param demand_factor: Adjustment based on demand (default is 1.0).
    :param competition_factor: Adjustment based on competition pricing (default is 1.0).
    :return: Adjusted price.
    """
    adjusted_price = current_price * real_time_factor * demand_factor * competition_factor
    return round(adjusted_price, 2)

def dynamic_pricing_analysis(product_name):
    """
    Provides dynamic pricing analysis based on historical sales data, competition, and demand.
    :param product_name: The name of the product to analyze.
    :return: A tuple of recommended dynamic price, price elasticity, and visualization.
    """
    product_data = df[df["product_name"] == product_name]
    product_data['price_change'] = product_data['base_price'].pct_change().fillna(0)
    product_data['sales_growth'] = product_data['sales_volume'].pct_change().fillna(0)
    price_elasticity = product_data['sales_growth'].mean() / product_data['price_change'].mean() if product_data['price_change'].mean() != 0 else 0
    avg_sales = product_data['sales_volume'].mean()
    avg_price = product_data['base_price'].mean()
    demand_factor = 1 + (avg_sales / 1000)
    competition_factor = 1.1
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Volume', color='tab:blue')
    ax1.plot(product_data['date'], product_data['sales_volume'], color='tab:blue', label="Sales Volume")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Price', color='tab:orange')
    ax2.plot(product_data['date'], product_data['base_price'], color='tab:orange', label="Base Price")
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    fig.tight_layout()
    st.pyplot(fig)
    current_price = product_data['base_price'].mean()
    adjusted_price = penyesuaian_harga_dinamis(current_price, real_time_factor=1.0, demand_factor=demand_factor, competition_factor=competition_factor)
    return adjusted_price, price_elasticity

def evaluasi_elastisitas(product_name):
    """
    Evaluates price elasticity based on historical sales and price data.
    Elasticity is calculated as the percentage change in sales divided by the percentage change in price.
    :param product_name: The product for which elasticity is to be evaluated.
    :return: Elasticity value.
    """
    product_data = df[df["product_name"] == product_name]
    product_data['price_change'] = product_data['base_price'].pct_change().fillna(0)
    product_data['sales_change'] = product_data['sales_volume'].pct_change().fillna(0)
    price_elasticity = (product_data['sales_change'] / product_data['price_change']).mean()
    return round(price_elasticity, 2)

def optimasi_profit(product_name, cost):
    kategori = df[df["product_name"] == product_name]["category"].iloc[0]
    now = datetime.datetime.now()
    pred_price, some_other_value, another_value, yet_another_value = rekomendasi_harga(
        product_name,
        now.month,
        now.year,
        kategori,
        "Supermarket",
        "Retail"
    )
    if isinstance(pred_price, tuple):
        pred_price = pred_price[0]
    elastisitas = evaluasi_elastisitas(product_name)
    margin = (pred_price - cost) / pred_price * 100
    if elastisitas < 0:
        margin_adjustment = 5
        margin -= margin_adjustment
    return round(margin, 2), elastisitas

def hitung_break_even(cost, fixed_cost, volume):
    if volume == 0:
        return np.nan
    return round((fixed_cost + cost * volume) / volume, 2)

def analisis_performa_produk():
    performance = df.groupby("product_name")["sales_volume"].sum().reset_index()
    high_performance = performance[performance["sales_volume"] > performance["sales_volume"].mean()]
    low_performance = performance[performance["sales_volume"] <= performance["sales_volume"].mean()]
    return high_performance, low_performance

def plot_performance_analysis(high_perf, low_perf):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(high_perf["product_name"], high_perf["sales_volume"], color='green', label="High Performance", s=100)
    ax.scatter(low_perf["product_name"], low_perf["sales_volume"], color='red', label="Low Performance", s=100)
    ax.set_xlabel("Product Name")
    ax.set_ylabel("Sales Volume")
    ax.set_title("High vs Low Performance Products")
    ax.legend()
    plt.xticks(rotation=90)
    st.pyplot(fig)
    
def plot_break_even(cost, fixed_cost, volume):
    volumes = np.linspace(1, 200, 100)
    break_even_prices = [(fixed_cost + cost * v) / v for v in volumes]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(volumes, break_even_prices, label="Break Even Price", color='blue', linewidth=2)
    ax.axvline(x=volume, color='red', linestyle='--', label=f"Current Volume: {volume} units")
    ax.axhline(y=break_even, color='green', linestyle='--', label=f"Break Even Price: Rp{break_even}")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Break Even Price (Rp)")
    ax.set_title("Break Even Analysis")
    ax.legend()
    st.pyplot(fig)

def clustering_produk(n_clusters=4):
    """
    Perform clustering on the product data based on sales and price.
    :param n_clusters: The number of clusters to form (default is 4)
    :return: DataFrame with cluster labels
    """
    try:
        return segmentasi_pasar(n_clusters=n_clusters)
    except Exception as e:
        st.error(f"Error in clustering: {e}")
        return pd.DataFrame()

def hybrid_model_prediction(data_input):
    hybrid_pred = model_rf.predict(data_input)[0]
    return round(hybrid_pred, 2)

def generate_report(df):
    avg_sales_per_category = df.groupby('category')['avg_sales_per_month'].mean().reset_index()
    avg_price_per_category = df.groupby('category')['avg_price_per_month'].mean().reset_index()
    total_sales_per_year = df.groupby('year')['avg_sales_per_month'].sum().reset_index()
    total_price_per_year = df.groupby('year')['avg_price_per_month'].sum().reset_index()
    category_report = pd.merge(avg_sales_per_category, avg_price_per_category, on='category', how='outer')
    year_report = pd.merge(total_sales_per_year, total_price_per_year, on='year', how='outer')
    return category_report, year_report

def notify_optimal_price_api(product_name, final_price):
    payload = {
        "product": product_name,
        "optimal_price": final_price,
        "timestamp": datetime.datetime.now().isoformat()
    }
    return f"Notifikasi API: {payload}"

def analisis_harga_kanal(product_name, channel):
    harga_produk = df[(df["product_name"] == product_name) & (df["channel"] == channel)]["base_price"].mean()
    harga_kompetitor = analisis_kompetitor(product_name)
    return {
        "harga_produk": round(harga_produk, 2),
        "harga_kompetitor": round(harga_kompetitor, 2),
        "selisih": round(harga_produk - harga_kompetitor, 2)
    }
    
def pricing_loyalty(product_name, customer_loyalty_level):
    pred_price, strategi, final_price, penjelasan = rekomendasi_harga(
        product_name,
        datetime.datetime.now().month,
        datetime.datetime.now().year,
        df[df["product_name"] == product_name]["category"].iloc[0],
        "Supermarket",
        "Retail"
    )
    if pred_price is None:
        return None, None, None, None
    if customer_loyalty_level == "Gold":
        final_price *= 0.90 
    elif customer_loyalty_level == "Silver":
        final_price *= 0.95
    return round(final_price, 2), strategi, round(final_price, 2), penjelasan

def dynamic_pricing_by_time(product_name):
    now = datetime.datetime.now()
    current_hour = now.hour
    kategori = df[df["product_name"] == product_name]["category"].iloc[0]
    pred_price, _, _ = rekomendasi_harga(
        product_name,
        now.month,
        now.year,
        kategori,
        "Supermarket",
        "Retail"
    )
    if 18 <= current_hour <= 21:
        final_price = pred_price * 1.05
    elif current_hour < 9 or current_hour > 22:
        final_price = pred_price * 0.95
    else:
        final_price = pred_price
    return round(final_price, 2)

def analisis_promo(product_name):
    cutoff_date = df["date"].max() - pd.Timedelta(days=30)
    before = df[(df["date"] < cutoff_date) & (df["product_name"] == product_name)]["sales_volume"].mean()
    after = df[(df["date"] >= cutoff_date) & (df["product_name"] == product_name)]["sales_volume"].mean()
    return {
        "before_promo": round(before, 2),
        "after_promo": round(after, 2),
        "delta": round(after - before, 2)
    }

def visualize_pricing_dashboard(df):
    st.subheader("Distribusi Harga Produk & Boxplot Harga Produk")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["base_price"], bins=30, color='skyblue', edgecolor='black')
        ax.set_title("Distribusi Harga Produk")
        ax.set_xlabel("Harga")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df["base_price"], color='lightgreen', ax=ax)
        ax.set_title("Boxplot Harga Produk")
        ax.set_xlabel("Harga")
        st.pyplot(fig)
    st.subheader("Rata-rata Harga & Penjualan per Kategori")
    col3, col4 = st.columns(2)
    with col3:
        avg_price_per_category = df.groupby("category")["avg_price_per_month"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="category", y="avg_price_per_month", data=avg_price_per_category, ax=ax, palette='muted')
        ax.set_title("Rata-rata Harga per Kategori")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Harga Rata-rata per Bulan")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col4:
        avg_sales_per_category = df.groupby("category")["avg_sales_per_month"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="category", y="avg_sales_per_month", data=avg_sales_per_category, ax=ax, palette='viridis')
        ax.set_title("Rata-rata Penjualan per Kategori")
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Penjualan Rata-rata per Bulan")
        plt.xticks(rotation=45)
        st.pyplot(fig)

def visualize_sales_trend(product_name):
    product_data = df[df["product_name"] == product_name].sort_values("date")
    start_date, end_date = st.slider(
        "Pilih Rentang Tanggal:",
        min_value=product_data["date"].min().date(),
        max_value=product_data["date"].max().date(),
        value=(product_data["date"].min().date(), product_data["date"].max().date())
    )
    product_data = product_data[(product_data["date"] >= pd.to_datetime(start_date)) & 
                                 (product_data["date"] <= pd.to_datetime(end_date))]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=product_data, x="date", y="sales_volume", marker='o', color='green', ax=ax)
    product_data['moving_avg'] = product_data['sales_volume'].rolling(window=7).mean()
    sns.lineplot(data=product_data, x="date", y="moving_avg", color='orange', ax=ax, label="Moving Average")
    ax.set_title(f"Tren Penjualan - {product_name}", fontsize=16)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Volume Penjualan", fontsize=12)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

def adaptive_learning_update_model(new_data):
    global model_rf, X_train, y_train
    try:
        X_new = new_data[features]
        y_new = new_data[target]
    except Exception as e:
        st.error("Pastikan data baru memiliki kolom yang diperlukan.")
        logging.error("Error adaptive_learning_update_model: %s", e)
        return "Gagal memperbarui model."
    X_train_updated = pd.concat([X_train, X_new])
    y_train_updated = pd.concat([y_train, y_new])
    model_rf.fit(X_train_updated, y_train_updated)
    logging.info("Model telah diperbarui dengan data baru.")
    return "Model telah diperbarui dengan data baru."

def prediksi_jangka_panjang(product_name, years=3):
    kategori = df[df["product_name"] == product_name]["category"].iloc[0]
    rekomendasi_result = rekomendasi_harga(
        product_name,
        datetime.datetime.now().month,
        datetime.datetime.now().year,
        kategori,
        "Supermarket",
        "Retail"
    )
    pred_harga = rekomendasi_result[0]
    skenario = {}
    for i in range(1, years + 1):
        kenaikan = pred_harga * ((1 + 0.1) ** i)
        penurunan = pred_harga * ((1 - 0.1) ** i)
        skenario[f"Tahun_{i}"] = {
            "potensi_naik": round(kenaikan, 2),
            "potensi_turun": round(penurunan, 2)
        }
    return skenario

st.set_page_config(
    page_title="Sistem Prediksi Harga dan Strategi Penjualan Berdasarkan Pembelajaran Mesin - Fitri HY",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.header("Sistem Prediksi Harga dan Strategi Penjualan Berdasarkan Pembelajaran Mesin", divider=True)

st.sidebar.header("Navigasi Fitur")
pilihan = st.sidebar.radio("Pilih Fitur:", (
    "Evaluasi Model", 
    "Prediksi Harga & Rekomendasi", 
    "Simulasi Profit & ROI", 
    "Analisis Kompetitor", 
    "Forecast Penjualan", 
    "Segmentasi & Clustering", 
    "Deteksi Anomali", 
    "Rekomendasi Stok", 
    "Dynamic Pricing", 
    "Elastisitas Harga & Optimasi Profit", 
    "Break Even & Analisis Performa", 
    "Hybrid Model & Adaptive Learning", 
    "Analisis Promo & Channel", 
    "Pricing Loyalty", 
    "Prediksi Jangka Panjang & Risiko", 
    "Dashboard & Report", 
    "Visualisasi Tren Penjualan"
))

if pilihan == "Evaluasi Model":
    st.header("Evaluasi Model RandomForest")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    with col2:
        st.metric(label="R² Score", value=f"{r2:.2f}")
    with col3:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    st.subheader("Interpretasi Evaluasi Model")
    st.markdown("""
    - **Mean Absolute Error (MAE):** Menunjukkan rata-rata selisih absolut antara nilai yang diprediksi dan nilai sebenarnya.  
    - **R² Score:** Mengukur seberapa baik model menjelaskan variasi dalam data. Nilai mendekati 1 berarti model sangat baik.  
    - **Root Mean Squared Error (RMSE):** Mengukur rata-rata kesalahan kuadrat dan memberikan bobot lebih pada kesalahan besar.  
    """)
    col1, col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(6, 4))
        plt.plot(y_test, label="Nilai Sebenarnya", color="blue", linestyle='--')
        plt.plot(y_pred, label="Nilai Prediksi", color="red", alpha=0.7)
        plt.title("Prediksi vs Nilai Sebenarnya")
        plt.xlabel("Index")
        plt.ylabel("Nilai")
        plt.legend()
        st.pyplot(plt)
    with col2:
        plt.figure(figsize=(6, 4))
        residuals = y_test - y_pred
        sns.scatterplot(x=y_pred, y=residuals, color='green')
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("Nilai Prediksi")
        plt.ylabel("Residuals")
        st.pyplot(plt)
    col1, col2 = st.columns(2)
    with col1:
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=20, color="purple")
        plt.title("Distribusi Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frekuensi")
        st.pyplot(plt)
    with col2:
        plt.figure(figsize=(6, 4))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("QQ Plot Residuals")
        st.pyplot(plt)
    mse = mean_squared_error(y_test, y_pred)
    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    st.subheader("Nilai Sebenarnya vs Prediksi")
    results_df = pd.DataFrame({
        "Index": range(len(y_test)),
        "Nilai Sebenarnya": y_test,
        "Nilai Prediksi": y_pred,
        "Residuals": residuals
    })
    st.dataframe(results_df)
    st.markdown("""
    **Analisis Hasil Model**  
    - Jika residuals memiliki pola tertentu, pertimbangkan untuk meningkatkan model.  
    - Jika nilai R² sangat rendah, model perlu ditingkatkan atau fitur tambahan diperlukan.  
    - Jika histogram residuals tidak normal, model mungkin memiliki bias atau heteroskedastisitas.  
    """)

elif pilihan == "Prediksi Harga & Rekomendasi":
    st.header("Prediksi Harga & Rekomendasi AI", anchor="prediksi-harga")
    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("Pilih Produk:", df["product_name"].unique(), key="product_select")
        month = st.number_input("Bulan (1-12):", min_value=1, max_value=12, 
                                value=datetime.datetime.now().month, key="month_input")
        year = st.number_input("Tahun:", min_value=2000, max_value=2100, 
                               value=datetime.datetime.now().year, key="year_input")
        kategori = st.selectbox("Pilih Kategori Produk:", df["category"].unique(), key="kategori_select")
    with col2:
        region = st.selectbox("Pilih Wilayah:", ("Default", "Urban", "Rural"), key="region_select")
        customer_segment = st.selectbox("Segmentasi Pelanggan:", df["customer_segment"].unique(), key="segment_select")
        channel = st.selectbox("Kanal Distribusi:", df["channel"].unique(), key="channel_select")
    st.write(f"Kategori Produk yang Dipilih: **{kategori}**")
    pred_harga, strategi, final_harga, penjelasan = rekomendasi_harga(product, month, year, kategori, customer_segment, channel, region)
    if pred_harga is not None:
        st.subheader("Hasil Prediksi Harga & Rekomendasi")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Produk", f"{product}")
        with col4:
            st.metric("Prediksi Harga", f"Rp{pred_harga:,}")
        with col5:
            st.metric("Harga Final", f"Rp{final_harga:,}")
        st.markdown(f"Strategi Harga: \n> **{strategi}**")
        st.subheader("Penjelasan")
        st.write(penjelasan)
        if st.button("Notifikasi Harga Optimal"):
            notifikasi = notify_optimal_price_api(product, final_harga)
            st.success(notifikasi)

elif pilihan == "Simulasi Profit & ROI":
    st.header("Simulasi Profit & ROI")
    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    with col2:
        cost = st.number_input("Masukkan Biaya (Cost):", value=10000.0, min_value=0.0, step=100.0, format="%.2f")
    with col3:
        volume = st.number_input("Volume Penjualan:", min_value=1, value=100, step=10)
    kategori = df[df["product_name"] == product]["category"].iloc[0]
    pred_harga, strategi, final_harga, penjelasan = rekomendasi_harga(
        product, datetime.datetime.now().month, datetime.datetime.now().year,
        kategori, "Supermarket", "Retail"
    )
    if final_harga is not None:
        profit, roi = simulasi_profit_roi(final_harga, cost, volume)
        st.subheader("Hasil Simulasi")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimasi Profit", f"Rp {profit:,.2f}")
        with col2:
            st.metric("ROI", f"{roi:.2f} %")
        with col3:
            st.metric("Harga Final", f"Rp {final_harga:,.2f}")
        st.subheader("Strategi Harga")
        st.markdown(f"""
            {strategi}
        """)
        st.subheader("Penjelasan Rekomendasi Harga")
        st.markdown(f"""
            {penjelasan}
        """)

elif pilihan == "Analisis Kompetitor":
    st.header("Analisis Harga Kompetitor & Prediksi Harga Saing")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    comp_price = analisis_kompetitor(product)
    pred_price = prediksi_harga_saing(
        product, datetime.datetime.now().month, datetime.datetime.now().year,
        df[df["product_name"] == product]["category"].iloc[0], "Supermarket", "Retail"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Harga Rata-rata Kompetitor ({product})", value=f"Rp{comp_price:,.2f}")
    with col2:
        st.metric(label=f"Harga Prediksi ({product})", value=f"Rp{pred_price:,.2f}")
    st.subheader("Perbandingan Harga")
    plot_comparison(product, pred_price, comp_price)
    product_data = df[df["product_name"] == product]
    harga_produk = product_data["base_price"].mean()
    if pred_price > comp_price:
        strategi = "Harga Anda lebih tinggi daripada kompetitor. Pertimbangkan untuk menurunkan harga sedikit agar lebih kompetitif."
    elif pred_price < comp_price:
        strategi = "Harga Anda lebih rendah daripada kompetitor. Anda bisa mempertahankan harga atau meningkatkan sedikit jika ada permintaan tinggi."
    else:
        strategi = "Harga Anda setara dengan kompetitor. Pertahankan strategi harga saat ini."
    st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 5px solid {'#dc3545' if pred_price > comp_price else '#28a745' if pred_price < comp_price else '#ffc107'};">
            {strategi}
        </div>
    """, unsafe_allow_html=True)

elif pilihan == "Forecast Penjualan":
    st.header("Forecast Penjualan Masa Depan")
    periods = st.slider("Jumlah Periode Forecast:", 1, 12, 6)
    product_name = st.selectbox("Pilih Nama Produk:", df["product_name"].unique())
    forecast_df, rmse = forecast_penjualan(periods, product_name)
    if forecast_df is not None:
        st.subheader("Forecast Penjualan")
        st.write(forecast_df)
        st.write(f"RMSE (Root Mean Squared Error): **{rmse:.2f}**")
        
elif pilihan == "Segmentasi & Clustering":
    st.header("Segmentasi Pasar & Clustering Produk")
    if st.button("Show Elbow Method for Optimal Clusters"):
        plot_elbow_method()
    clusters = clustering_produk(n_clusters=4)
    if not clusters.empty:
        st.dataframe(clusters)
    else:
        st.warning("No data to display. Please check the data source.")

elif pilihan == "Deteksi Anomali":
    st.header("Deteksi Anomali Harga")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    threshold = st.slider("Threshold (z-score):", 1, 5, 2)
    anomalies = deteksi_anomali(product, threshold)
    if not anomalies.empty:
        st.write("**Harga Anomali:**")
        st.write(anomalies)
        st.download_button(
            label="Download Anomalies Data",
            data=anomalies.to_csv(index=False),
            file_name=f"{product}_anomalies.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada anomali terdeteksi.")

elif pilihan == "Rekomendasi Stok":
    st.header("Rekomendasi Stok Berdasarkan Tren Penjualan")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    rekomendasi, trend_message = rekomendasi_stok(product)
    st.markdown(f"""
        - Rekomendasi Stok: 
            > **{rekomendasi}**
        - Penjelasan Tren Penjualan: 
            > **{trend_message}**
    """)

elif pilihan == "Dynamic Pricing":
    st.header("Penyesuaian Harga Dinamis Real-Time")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    current_price = df[df["product_name"] == product]["base_price"].mean()
    st.metric("Harga Saat Ini", f"Rp{round(current_price,2):,}")
    real_time_factor = st.slider("Real-Time Factor:", 0.5, 1.5, 1.0, step=0.05)
    adjusted_price, price_elasticity = dynamic_pricing_analysis(product)
    st.markdown(f"""
    - Harga Setelah Penyesuaian Dinamis: **Rp{adjusted_price:,}**
    - Elastisitas Harga (Price Elasticity): **{round(price_elasticity, 2)}**
    > Elastisitas harga menunjukkan bagaimana penjualan dipengaruhi oleh perubahan harga. Nilai negatif berarti penurunan harga dapat meningkatkan penjualan.
    """)
    
elif pilihan == "Elastisitas Harga & Optimasi Profit":
    st.header("Evaluasi Elastisitas Harga & Optimasi Profit")
    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    with col2:
        cost = st.number_input("Masukkan biaya (cost):", value=10000.0)
    elastisitas = evaluasi_elastisitas(product)
    margin, elasticity = optimasi_profit(product, cost)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Elastisitas Harga", f"{elastisitas}")
    with col2:
        st.metric("Margin Ideal (Optimasi Profit)", f"{margin}%")
    if elasticity < 0:
        st.write("Penurunan harga mungkin meningkatkan penjualan berdasarkan elastisitas harga negatif.")
    elif elasticity > 0:
        st.write("Kenaikan harga dapat meningkatkan pendapatan, tetapi dapat menurunkan penjualan.")
    else:
        st.write("Elastisitas harga mendekati nol, yang menunjukkan bahwa harga tidak terlalu mempengaruhi penjualan.")
    st.write("Optimasi harga dan margin didasarkan pada elastisitas harga dan analisis pasar.")
    price_range = np.linspace(cost * 0.5, cost * 2, 100)
    profit_margins = [optimasi_profit(product, price)[0] for price in price_range]
    fig, ax = plt.subplots()
    ax.plot(price_range, profit_margins, label='Profit Margin', color='blue')
    ax.set_xlabel("Price")
    ax.set_ylabel("Profit Margin (%)")
    ax.set_title("Price Elasticity & Profit Margin Optimization")
    ax.legend()
    st.pyplot(fig)

elif pilihan == "Break Even & Analisis Performa":
    st.header("Kalkulasi Break-Even & Analisis Performa Produk")
    col1, col2, col3 = st.columns(3)
    with col1:
        cost = st.number_input("Masukkan biaya (cost):", value=10000.0)
    with col2:
        fixed_cost = st.number_input("Masukkan Fixed Cost:", value=50000.0)
    with col3:
        volume = st.number_input("Volume:", min_value=1, value=100)
    break_even = hitung_break_even(cost, fixed_cost, volume)
    st.metric(label="Break Even Price", value=f"Rp{break_even:,}")
    high_perf, low_perf = analisis_performa_produk()
    st.subheader("Produk High Performance")
    st.dataframe(high_perf)
    st.subheader("Produk Low Performance")
    st.dataframe(low_perf)
    col4, col5 = st.columns(2)
    with col4:
        st.write("Break Even Chart")
        plot_break_even(cost, fixed_cost, volume)
    with col5:
        st.write("High vs Low Sales Volume")
        plot_performance_analysis(high_perf, low_perf)
    
elif pilihan == "Hybrid Model & Adaptive Learning":
    st.header("Prediksi Hybrid Model & Adaptive Learning")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    st.metric(label="Produk yang dipilih", value=f"{product}")
    filtered_data = df[df["product_name"] == product]
    if filtered_data.empty:
        st.warning("Data untuk produk ini tidak ditemukan.")
    else:
        st.write("Data yang difilter untuk produk:", filtered_data)
    kategori = filtered_data["category"].iloc[0] if not filtered_data.empty else None
    data_input = pd.DataFrame({
        "month": [datetime.datetime.now().month],
        "year": [datetime.datetime.now().year],
        "product_encoded": [le_product.transform([product])[0]],
        "category_encoded": [le_category.transform([kategori])[0]] if kategori else [None],
        "customer_encoded": [le_customer.transform(["Supermarket"])[0]],
        "channel_encoded": [le_channel.transform(["Retail"])[0]],
        "avg_price_per_month": [filtered_data["base_price"].mean()],
        "avg_sales_per_month": [filtered_data["sales_volume"].mean()]
    })
    st.write("Data input untuk prediksi:", data_input)
    hybrid_pred = hybrid_model_prediction(data_input)
    st.write(f"Prediksi Hybrid Model untuk Produk {product}: **Rp{hybrid_pred:,}")
    product_data_cat = filtered_data[['product_name', 'category']].drop_duplicates()
    st.write("Data kategori produk:", product_data_cat)
    product_data_num = filtered_data.agg({
        'base_price': 'mean',
        'sales_volume': 'mean'
    }).reset_index()
    st.write("Data numerik untuk produk:", product_data_num)
    product_data = product_data_cat.merge(product_data_num, left_index=True, right_index=True)
    st.write("Data yang telah digabungkan:", product_data)
    product_data["predicted_sales"] = hybrid_pred
    st.subheader("Data Produk dan Prediksi")
    st.write(product_data)
    st.subheader("Update Model dengan Data Baru")
    new_data = filtered_data
    if not new_data.empty:
        msg = adaptive_learning_update_model(new_data)
        st.success(msg)
    else:
        st.warning("Tidak ada data baru untuk produk ini untuk memperbarui model.")

elif pilihan == "Analisis Promo & Channel":
    st.header("Analisis Efektivitas Promo & Analisis Harga Berdasarkan Kanal")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    promo = analisis_promo(product)
    st.subheader("Analisis Promo")
    promo_data = {
        "Before Promo (Avg Sales)": promo["before_promo"],
        "After Promo (Avg Sales)": promo["after_promo"],
        "Delta (Change in Sales)": promo["delta"]
    }
    st.table(promo_data)
    channel = st.selectbox("Pilih Saluran Penjualan:", df["channel"].unique())
    channel_analysis = analisis_harga_kanal(product, channel)
    st.subheader("Analisis Harga Berdasarkan Kanal")
    channel_data = {
        "Harga Produk (Rp)": channel_analysis["harga_produk"],
        "Harga Kompetitor (Rp)": channel_analysis["harga_kompetitor"],
        "Selisih Harga (Rp)": channel_analysis["selisih"]
    }
    st.table(channel_data)

elif pilihan == "Pricing Loyalty":
    st.header("Strategi Harga Berdasarkan Customer Loyalty")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    loyalty = st.selectbox("Tingkat Loyalitas:", ("Bronze", "Silver", "Gold"))
    final_price, strategi, adjusted_price, penjelasan = pricing_loyalty(product, loyalty)
    if final_price is not None:
        st.subheader("Detail Harga")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Produk", value=f"{product}")
            st.metric(label="Tingkat Loyalitas", value=f"{loyalty}")
        with col2:
            st.metric(label="Harga Prediksi", value=f"Rp{final_price:,}")
            st.metric(label="Direkomendasikan", value=f"Rp{adjusted_price:,}")
        st.write(f"Strategi: \n > {strategi}")
        st.subheader("Penjelasan")
        st.markdown(f"{penjelasan}")
    else:
        st.error("Tidak dapat menghitung harga yang disesuaikan. Periksa data atau pilih produk yang valid.")

elif pilihan == "Prediksi Jangka Panjang & Risiko":
    st.header("Prediksi Harga Jangka Panjang & Simulasi Risiko")
    product = st.selectbox("Pilih Produk:", df["product_name"].unique())
    years = st.slider("Prediksi untuk berapa tahun ke depan:", 1, 5, 3)
    risiko = prediksi_jangka_panjang(product, years)
    if risiko:
        st.subheader("Simulasi Risiko Harga")
        pred_harga, strategi, adjusted_price, penjelasan = rekomendasi_harga(
            product,
            datetime.datetime.now().month,
            datetime.datetime.now().year,
            df[df["product_name"] == product]["category"].iloc[0],
            "Supermarket",
            "Retail"
        )
        pred_harga = f"Rp{pred_harga:,.2f}"
        adjusted_price = f"Rp{adjusted_price:,.2f}"
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Produk", value=f"{product}")
        with col2:
            st.metric(label="Harga Prediksi", value=f"{pred_harga}")
        with col3:
            st.metric(label="Direkomendasikan", value=f"{adjusted_price}")
        st.write(f"Strategi Penyesuaian Harga: \n > {strategi}")
        st.write(f"{penjelasan}")
        st.write(f"### Periode Prediksi: {years} Tahun ke Depan")
        for year, scenario in risiko.items():
            st.write(f"##### {year}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Potensi Kenaikan Harga", value=f"Rp{scenario['potensi_naik']:,.2f}")
            with col2:
                st.metric(label="Potensi Penurunan Harga", value=f"Rp{scenario['potensi_turun']:,.2f}")
    else:
        st.error("Tidak dapat menghasilkan prediksi harga jangka panjang. Periksa data produk atau coba lagi.")
 
elif pilihan == "Dashboard & Report":
    st.header("Dashboard & Report")
    df = load_data()
    if df.empty:
        st.error("Data tidak ditemukan. Pastikan file data tersedia.")
    else:
        visualize_pricing_dashboard(df)
        if st.button("Generate Laporan Otomatis"):
            category_report, year_report = generate_report(df)
            st.subheader("Laporan per Kategori")
            st.dataframe(category_report)
            st.subheader("Laporan per Tahun")
            st.dataframe(year_report)

elif pilihan == "Visualisasi Tren Penjualan":
    st.header("Visualisasi Tren Penjualan Produk")
    product = st.selectbox("Pilih Produk untuk Tren Penjualan:", df["product_name"].unique())
    visualize_sales_trend(product)

st.markdown("---")
st.caption("© 2025 Sistem Prediksi Harga & Analisis Penjualan Berbasis AI")
