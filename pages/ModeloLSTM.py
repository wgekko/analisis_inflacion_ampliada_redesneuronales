# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import warnings
from datetime import datetime, timedelta

# Configuración básica
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC - LSTM & Analytics", layout="wide")

st.header(":material/neurology: Proyección de Inflación : IPC (Modelo LSTM) & Analytics")

# --- 1. CARGA DE DATOS (CACHED) ---
@st.cache_data
def cargar_y_limpiar_datos(file_path):
    if not os.path.exists(file_path):
        return None, None

    try:
        raw_df = pd.read_excel(file_path, sheet_name='Índices aperturas', header=None, dtype=str)
    except:
        raw_df = pd.read_excel(file_path, header=None, dtype=str)

    mask = raw_df.astype(str).apply(lambda r: r.str.contains('2016-12', na=False).any(), axis=1)
    matching_indices = raw_df.index[mask].tolist()

    if not matching_indices:
        return None, None
        
    date_row_idx = matching_indices[0]
    dates_raw = raw_df.iloc[date_row_idx, 1:].replace('nan', np.nan).dropna().values
    dates_formatted = pd.to_datetime(dates_raw).strftime('%Y-%m')
    
    cleaned_data = []
    current_region = "Sin Región"

    for idx, row in raw_df.iloc[date_row_idx:].iterrows():
        first_col = str(row[0]).strip()
        if pd.isna(first_col) or first_col == "" or "nan" in first_col.lower():
            continue
            
        if "región" in first_col.lower():
            current_region = first_col
        else:
            item_name = first_col
            values_raw = row[1:len(dates_raw)+1].values
            values_clean = pd.to_numeric(pd.Series(values_raw).replace(['///', 'nan', ''], np.nan), errors='coerce')
            
            if values_clean.notnull().sum() > 30:
                values_clean = values_clean.interpolate(method='linear').bfill().ffill().values
                cleaned_data.append({
                    'Region': current_region,
                    'Item': item_name,
                    'Values': values_clean
                })
    
    return cleaned_data, dates_formatted

# --- 2. MODELO ---
def entrenar_y_predecir_lstm(valores, seq_length=24, epochs=30, steps=3):
    if len(valores) <= seq_length: return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(valores.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(seq_length, len(scaled_data)):
        x_train.append(scaled_data[x-seq_length:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=64),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=epochs, batch_size=16, verbose=0)

    current_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    predictions_scaled = []
    for _ in range(steps):
        pred = model.predict(current_seq, verbose=0)
        predictions_scaled.append(pred[0][0])
        current_seq = np.append(current_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)
    
    return scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# --- 3. LÓGICA PRINCIPAL ---
FILE_PATH = "data/ipc.xlsx"
data_list, historical_dates = cargar_y_limpiar_datos(FILE_PATH)

if data_list:
    st.sidebar.header(":material/settings: Configuración")
    regiones = sorted(list(set([d['Region'] for d in data_list])))
    region_sel = st.sidebar.selectbox("Seleccione Región", ["Todas"] + regiones)
    
    if st.button("Iniciar Entrenamiento y Proyección"):
        procesar = data_list if region_sel == "Todas" else [d for d in data_list if d['Region'] == region_sel]
        
        res_3m = []
        progreso = st.progress(0)
        status = st.empty()
        
        for i, entry in enumerate(procesar):
            status.text(f"Entrenando {i+1}/{len(procesar)}: {entry['Item']}")
            preds = entrenar_y_predecir_lstm(entry['Values'])
            if preds is not None:
                ultimo = entry['Values'][-1]
                p1, p2, p3 = preds[0], preds[1], preds[2]
                v1, v2, v3 = ((p1/ultimo)-1)*100, ((p2/p1)-1)*100, ((p3/p2)-1)*100
                
                # Calcular última variación histórica real para la tabla
                penultimo = entry['Values'][-2] if len(entry['Values']) > 1 else ultimo
                u_pct = ((ultimo / penultimo) - 1) * 100 if penultimo != 0 else 0
                
                res_3m.append({
                    "Región": entry['Region'], 
                    "Ítem": entry['Item'],
                    "Último Dato (%)": round(u_pct, 2), 
                    "Mes 1 Proyectado (%)": round(v1, 2),
                    "Mes 2 Proyectado (%)": round(v2, 2),
                    "Mes 3 Proyectado (%)": round(v3, 2)
                })
            progreso.progress((i + 1) / len(procesar))
        
        status.success("¡Proyección completada!")
        st.session_state['df_resultados'] = pd.DataFrame(res_3m)

    # --- 4. ÁREA DE VISUALIZACIÓN ---
    if 'df_resultados' in st.session_state:
        df3 = st.session_state['df_resultados']
        
        st.subheader("Resultados de la Proyección (Variaciones Porcentuales)")
        st.dataframe(df3, width='stretch', hide_index=True)
        
        st.divider()
        st.subheader("Análisis de Variación Mensual (%)")
        
        item_graf = st.selectbox("Seleccione rubro para analizar la tasa de variación:", df3['Ítem'].unique())
        
        datos_item = next(d for d in data_list if d['Item'] == item_graf)
        val_hist_indices = datos_item['Values'][-25:] 
        fec_hist_completa = historical_dates[-25:]
        
        serie_indices = pd.Series(val_hist_indices)
        var_hist_pct = (serie_indices.pct_change() * 100).dropna().values
        fec_hist_grafico = fec_hist_completa[1:] 
        
        row_p = df3[df3['Ítem'] == item_graf].iloc[0]
        preds_pct = [row_p['Mes 1 Proyectado (%)'], row_p['Mes 2 Proyectado (%)'], row_p['Mes 3 Proyectado (%)']]
        
        try:
            u_fec = datetime.strptime(fec_hist_grafico[-1], '%Y-%m')
            fec_fut = [(u_fec + timedelta(days=31*i)).strftime('%Y-%m') for i in range(1, 4)]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fec_hist_grafico, 
                y=var_hist_pct, 
                name='Variación Real (%)', 
                line=dict(color='#2ca02c', width=3),
                mode='lines+markers'
            ))
            
            x_pred = [fec_hist_grafico[-1]] + fec_fut
            y_pred = [var_hist_pct[-1]] + preds_pct
            
            fig.add_trace(go.Scatter(
                x=x_pred, 
                y=y_pred, 
                name='Proyección de Tasa (%)', 
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f"Tasa de Inflación Mensual: {item_graf}",
                xaxis_title="Meses",
                yaxis_title="Variación Porcentual (%)",
                hovermode="x unified",
                template="plotly_white",
                yaxis=dict(ticksuffix="%") 
            )
            
            st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.error(f"Error al calcular variaciones: {e}")
else:
    st.error("No se pudo cargar el archivo 'data/ipc.xlsx'.")