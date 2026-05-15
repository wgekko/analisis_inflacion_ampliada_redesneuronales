import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import warnings
from datetime import datetime, timedelta

# Configuración y estética
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Predicción IPC-Modelo TCN & Analytics", layout="wide")

st.header(":material/neurology: Proyección de Inflación: Modelo TCN (Temporal Convolutional Network) & Analytics")

st.info("""
**Arquitectura TCN:** Este modelo utiliza convoluciones 1D con dilatación para captar dependencias de largo plazo en la tasa de variación de precios. 
Es más estable ante shocks económicos que los modelos tradicionales.
""")

@st.cache_data
def cargar_datos_ipc(file_path):
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
    fechas_dt = pd.to_datetime(fechas_raw).strftime('%Y-%m')
    
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

def build_tcn_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=1, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=32, kernel_size=3, padding='causal', dilation_rate=2, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=16, kernel_size=3, padding='causal', dilation_rate=4, activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    return model

def entrenar_y_predecir_tcn(serie, seq_length=24, steps=3):
    if len(serie) <= seq_length: return None
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(serie.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_tcn_model((seq_length, 1))
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)

    input_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    preds_scaled = []
    for _ in range(steps):
        p = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(p)
        new_val = np.array([[[p]]])
        input_seq = np.append(input_seq[:, 1:, :], new_val, axis=1)
    
    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# --- FLUJO PRINCIPAL ---
FILE_PATH = "data/ipc.xlsx"
data_list, dates_hist = cargar_datos_ipc(FILE_PATH)

if data_list:
    st.sidebar.header("Configuración")
    region_sel = st.sidebar.selectbox("Región de Análisis", sorted(list(set([d['Region'] for d in data_list]))))
    
    if st.button("Ejecutar Modelo TCN"):
        procesar = [d for d in data_list if d['Region'] == region_sel]
        resultados = []
        progreso = st.progress(0)
        status = st.empty()
        
        for i, entry in enumerate(procesar):
            status.text(f"Analizando: {entry['Item']}")
            preds = entrenar_y_predecir_tcn(entry['Values'])
            
            if preds is not None:
                resultados.append({
                    "Ítem": entry['Item'],
                    "Último Dato (%)": round(entry['Values'][-1], 2),
                    "Mes 1 Proyectado (%)": round(preds[0], 2),
                    "Mes 2 Proyectado (%)": round(preds[1], 2),
                    "Mes 3 Proyectado (%)": round(preds[2], 2),
                    "Tendencia": "⬆️ Suba" if preds[0] > entry['Values'][-1] else "⬇️ Baja"
                })
            progreso.progress((i + 1) / len(procesar))
        
        st.session_state['res_tcn'] = pd.DataFrame(resultados)
        status.success(f"Análisis TCN finalizado para {region_sel}")

    if 'res_tcn' in st.session_state:
        df_res = st.session_state['res_tcn']
        
        st.subheader(f"Predicciones de Tasa Mensual - {region_sel}")
        st.dataframe(df_res, width='stretch', hide_index=True)

        st.divider()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            item_graf = st.selectbox("Seleccione rubro para visualizar:", df_res['Ítem'].unique())
        
        with col2:
            entry = next(d for d in data_list if d['Item'] == item_graf and d['Region'] == region_sel)
            hist_vals = entry['Values'][-24:] 
            hist_dates = dates_hist[-24:]
            
            row_p = df_res[df_res['Ítem'] == item_graf].iloc[0]
            preds_vals = [row_p['Mes 1 Proyectado (%)'], row_p['Mes 2 Proyectado (%)'], row_p['Mes 3 Proyectado (%)']]
            
            ult_fec = datetime.strptime(hist_dates[-1], '%Y-%m')
            fut_dates = [(ult_fec + timedelta(days=31*(i+1))).strftime('%Y-%m') for i in range(3)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, name='Inflación Histórica (%)', 
                                    line=dict(color='#1f77b4', width=3), mode='lines+markers'))
            
            x_pred = [hist_dates[-1]] + fut_dates
            y_pred = [hist_vals[-1]] + preds_vals
            fig.add_trace(go.Scatter(x=x_pred, y=y_pred, name='Proyección TCN (%)', 
                                    line=dict(color='#d62728', width=3, dash='dash'), mode='lines+markers'))
            
            fig.update_layout(title=f"Evolución y Predicción: {item_graf}", 
                            yaxis_title="Variación Mensual (%)", template="plotly_dark",
                            hovermode="x unified", height=450,
                            yaxis=dict(ticksuffix="%"))
            st.plotly_chart(fig, width='stretch')
else:
    st.error("Por favor, asegúrate de que el archivo 'ipc.xlsx' esté en la carpeta 'data/'.")