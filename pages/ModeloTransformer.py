# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
from datetime import datetime, timedelta

# Configuración y estética
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC-Modelo Transformer", layout="wide")

st.header(":material/neurology: Proyección de Inflación: Modelo Transformer & Analytics ")

# Modificado: Ahora el caché depende también de la fecha de modificación del archivo
@st.cache_data
def cargar_datos_ipc(file_path, file_mtime):
    if not os.path.exists(file_path):
        return None, None
    try:
        raw_df = pd.read_excel(file_path, sheet_name='Índices aperturas', header=None, dtype=str)
    except:
        raw_df = pd.read_excel(file_path, header=None, dtype=str)

    mask = raw_df.astype(str).apply(lambda r: r.str.contains('2016-12', na=False).any(), axis=1)
    matching_indices = raw_df.index[mask].tolist()
    if not matching_indices: return None, None
        
    idx_fechas = matching_indices[0]
    fechas_raw = raw_df.iloc[idx_fechas, 1:].replace('nan', np.nan).dropna().values
    fechas_dt = pd.to_datetime(fechas_raw).strftime('%Y-%m').values # Convertimos a array de strings
    
    dataset = []
    region_actual = "Sin Región"

    for idx, row in raw_df.iloc[idx_fechas:].iterrows():
        item_col = str(row[0]).strip()
        if pd.isna(item_col) or item_col == "" or "nan" in item_col.lower(): continue
            
        if "región" in item_col.lower():
            region_actual = item_col
        else:
            valores_indices = pd.to_numeric(row[1:len(fechas_raw)+1].replace(['///', 'nan', ''], np.nan), errors='coerce')
            if valores_indices.notnull().sum() > 30:
                indices_ser = valores_indices.interpolate().bfill().ffill()
                variaciones = (indices_ser.pct_change() * 100).fillna(0).values
                dataset.append({
                    'Region': region_actual,
                    'Item': item_col,
                    'Values': variaciones 
                })
    
    return dataset, fechas_dt

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def entrenar_y_predecir_transformer(serie, seq_length=24, steps=3):
    if len(serie) <= seq_length: return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(serie.reshape(-1, 1))
    
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    model = build_transformer_model((seq_length, 1))
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    
    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    preds_scaled = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(p)
        new_val = np.array([[[p]]])
        input_seq = np.append(input_seq[:, 1:, :], new_val, axis=1)
    
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- EJECUCIÓN ---
FILE_PATH = "data/ipc.xlsx"

# Modificado: Detecta de forma dinámica si el archivo en el disco sufrió modificaciones
file_mtime = os.path.getmtime(FILE_PATH) if os.path.exists(FILE_PATH) else 0
data_list, dates_hist = cargar_datos_ipc(FILE_PATH, file_mtime)

if data_list is not None:
    st.sidebar.header("Configuración")

    region_sel = st.sidebar.selectbox("Región", sorted(list(set([d['Region'] for d in data_list]))))
    
    if st.button("Calcular Proyecciones Transformer"):
        procesar = [d for d in data_list if d['Region'] == region_sel]
        resultados = []
        bar = st.progress(0)
        for i, entry in enumerate(procesar):
            preds = entrenar_y_predecir_transformer(entry['Values'])
            if preds is not None:
                resultados.append({
                    "Ítem": entry['Item'],
                    "Último %": round(entry['Values'][-1], 2),
                    "Mes 1": round(preds[0], 2),
                    "Mes 2": round(preds[1], 2),
                    "Mes 3": round(preds[2], 2)
                })
            bar.progress((i + 1) / len(procesar))
        st.session_state['res_trans'] = pd.DataFrame(resultados)

    if 'res_trans' in st.session_state:
        df_res = st.session_state['res_trans']
        st.dataframe(df_res, width='stretch', hide_index=True)
        
        st.divider()
        st.subheader("Gráfico Dinámico con Zoom Temporal")
        item_graf = st.selectbox("Seleccione Rubro:", df_res['Ítem'].unique())
        
        # --- LÓGICA DE ZOOM ---
        entry_data = next(d for d in data_list if d['Item'] == item_graf and d['Region'] == region_sel)
        full_vals = entry_data['Values']
        full_dates = dates_hist
        
        rango = st.select_slider(
            "Deslice para ajustar el periodo visible:",
            options=list(full_dates),
            value=(full_dates[-24], full_dates[-1])
        )
        
        mask = (full_dates >= rango[0]) & (full_dates <= rango[1])
        vals_zoom = full_vals[mask]
        dates_zoom = full_dates[mask]
        
        row_p = df_res[df_res['Ítem'] == item_graf].iloc[0]
        preds_vals = [row_p['Mes 1'], row_p['Mes 2'], row_p['Mes 3']]
        
        # Modificado: Mensaje de depuración interno dinámico basado en las fechas reales
        ult_fec_dt = datetime.strptime(full_dates[-1], '%Y-%m')
        fut_dates = [(ult_fec_dt + timedelta(days=31*(i+1))).strftime('%Y-%m') for i in range(3)]
        
        # Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates_zoom, y=vals_zoom, name='Histórico', line=dict(color='#17BECF', width=3)))
        
        x_pred = [dates_zoom[-1]] + fut_dates
        y_pred = [vals_zoom[-1]] + preds_vals
        fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name='Predicción Transformer', 
                                line=dict(color='#FF4B4B', width=4, dash='dash'), mode='lines+markers'))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
else:
    st.error("Verifique que el archivo esté en data/ipc.xlsx")