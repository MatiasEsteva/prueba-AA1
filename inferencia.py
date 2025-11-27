"""
Script de inferencia para predicción de lluvia en Australia
Recibe input.csv y genera output.csv
"""

import pandas as pd
import numpy as np
import joblib
import sys
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, OneHotEncoder

class WeatherPreprocessor:
    """Encapsula todo el preprocesamiento personalizado"""
    
    def __init__(self):
        self.ratios = {}
        self.medianas_location_month = {}
        self.medianas_region_month = {}
        self.modas_region = {}
        self.scaler = None
        self.encoder = None
        self.wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
    
    def _impute_pair_cross(self, df, var1, var2, ratio_1_to_2, ratio_2_to_1):
        """Imputación cruzada entre pares"""
        df = df.copy()
        mask_missing_1 = df[var1].isna() & df[var2].notna()
        df.loc[mask_missing_1, var1] = df.loc[mask_missing_1, var2] * ratio_1_to_2
        
        mask_missing_2 = df[var2].isna() & df[var1].notna()
        df.loc[mask_missing_2, var2] = df.loc[mask_missing_2, var1] * ratio_2_to_1
        return df
    
    def transform(self, df):
        """Aplica todo el preprocesamiento"""
        df = df.copy()
        
        # 1. Imputaciones
        # Humedad
        df = self._impute_pair_cross(df, 'Humidity9am', 'Humidity3pm', 
                                     self.ratios['humidity_3pm_to_9am'], 
                                     self.ratios['humidity_9am_to_3pm'])
        for col in ['Humidity9am', 'Humidity3pm']:
            df[col] = df.apply(
                lambda row: self.medianas_location_month[col].get((row['Location'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Temp Min/Max
        df = self._impute_pair_cross(df, 'MinTemp', 'MaxTemp',
                                     self.ratios['temp_min_to_max'], self.ratios['temp_max_to_min'])
        for col in ['MinTemp', 'MaxTemp']:
            df[col] = df.apply(
                lambda row: self.medianas_location_month[col].get((row['Location'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Temp 9am/3pm
        df = self._impute_pair_cross(df, 'Temp9am', 'Temp3pm',
                                     self.ratios['temp_9am_to_3pm'], self.ratios['temp_3pm_to_9am'])
        for col in ['Temp9am', 'Temp3pm']:
            df[col] = df.apply(
                lambda row: self.medianas_location_month[col].get((row['Location'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Presión
        df = self._impute_pair_cross(df, 'Pressure9am', 'Pressure3pm',
                                     self.ratios['pressure_9am_to_3pm'], self.ratios['pressure_3pm_to_9am'])
        for col in ['Pressure9am', 'Pressure3pm']:
            df[col] = df.apply(
                lambda row: self.medianas_region_month[col].get((row['RegionCluster'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # WindSpeed
        for col in ['WindSpeed9am', 'WindSpeed3pm']:
            df[col] = df.apply(
                lambda row: self.medianas_location_month[col].get((row['Location'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Cloud
        df = df[~((df['Cloud9am'] == 9) | (df['Cloud3pm'] == 9))]
        df = self._impute_pair_cross(df, 'Cloud9am', 'Cloud3pm',
                                     self.ratios['cloud_9am_to_3pm'], self.ratios['cloud_3pm_to_9am'])
        for col in ['Cloud9am', 'Cloud3pm']:
            df[col] = df.apply(
                lambda row: self.medianas_region_month[col].get((row['RegionCluster'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # WindGustSpeed
        df['WindGustSpeed'] = df['WindGustSpeed'].fillna(df[['WindSpeed9am', 'WindSpeed3pm']].max(axis=1))
        
        # Rainfall
        df['Rainfall'] = df.apply(
            lambda row: self.medianas_location_month['Rainfall'].get((row['Location'], row['Month']), df['Rainfall'].median())
            if pd.isna(row['Rainfall']) else row['Rainfall'], axis=1
        )
        
        # Evaporation y Sunshine
        for col in ['Evaporation', 'Sunshine']:
            df[col] = df.apply(
                lambda row: self.medianas_region_month[col].get((row['RegionCluster'], row['Month']), df[col].median())
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Variables categóricas
        for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']:
            df[col] = df.apply(
                lambda row: self.modas_region[col].get(row['RegionCluster'], df[col].mode()[0])
                if pd.isna(row[col]) else row[col], axis=1
            )
        
        # Actualizar RainToday según Rainfall
        df.loc[df['Rainfall'] > 0, 'RainToday'] = 'Yes'
        df.loc[df['Rainfall'] == 0, 'RainToday'] = 'No'
        
        # Cloud a enteros
        df['Cloud9am'] = df['Cloud9am'].astype(int)
        df['Cloud3pm'] = df['Cloud3pm'].astype(int)
        
        # 2. Convertir RainToday a binario
        df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
        
        # 3. Codificar direcciones del viento (cíclicas)
        for col_base in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            df[col_base] = df[col_base].map(self.wind_dir_map)
            df[f'{col_base}_sin'] = np.sin(np.deg2rad(df[col_base]))
            df[f'{col_base}_cos'] = np.cos(np.deg2rad(df[col_base]))
        
        # 4. Crear diferencias
        df['temp_diff'] = abs(df['Temp9am'] - df['Temp3pm'])
        df['humidity_diff'] = abs(df['Humidity9am'] - df['Humidity3pm'])
        df['wind_diff'] = abs(df['WindSpeed9am'] - df['WindSpeed3pm'])
        df['min_max_diff'] = abs(df['MinTemp'] - df['MaxTemp'])
        df['cloud_diff'] = abs(df['Cloud9am'] - df['Cloud3pm'])
        df['pressure_diff'] = abs(df['Pressure9am'] - df['Pressure3pm'])
        
        # 5. One Hot Encoding
        encoded_array = self.encoder.transform(df[['Month', 'RegionCluster']])
        encoded_cols = self.encoder.get_feature_names_out(['Month', 'RegionCluster'])
        df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=['Month', 'RegionCluster']), df_encoded], axis=1)
        
        # 6. Eliminar columnas innecesarias
        cols_to_drop = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 
                       'Latitud', 'Longitud']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if 'RainTomorrow' in df.columns:
            cols_to_drop.append('RainTomorrow')
        df = df.drop(columns=cols_to_drop)
        
        # 7. Escalar
        features_escalar = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                           'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                           'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                           'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'temp_diff',
                           'humidity_diff', 'wind_diff', 'min_max_diff', 'cloud_diff', 'pressure_diff']
        
        df[features_escalar] = self.scaler.transform(df[features_escalar])
        
        return df


def assign_region_cluster(df):
    """
    Asigna cluster de región basado en coordenadas
    Asume que df ya tiene las columnas 'Latitud' y 'Longitud'
    """
    # NorfolkIsland
    df.loc[df['Location'] == 'NorfolkIsland', 'RegionCluster'] = 5
    
    # Continental Norte (latitud > -22)
    df.loc[(df['Location'] != 'NorfolkIsland') & (df['Latitud'] > -22), 'RegionCluster'] = 4
    
    # Continental Sur - división aproximada en 4 clusters
    mask_sur = (df['Location'] != 'NorfolkIsland') & (df['Latitud'] <= -22)
    df.loc[mask_sur & (df['Latitud'] > -30), 'RegionCluster'] = 0
    df.loc[mask_sur & (df['Latitud'] <= -30) & (df['Latitud'] > -35), 'RegionCluster'] = 1
    df.loc[mask_sur & (df['Latitud'] <= -35) & (df['Latitud'] > -37), 'RegionCluster'] = 2
    df.loc[mask_sur & (df['Latitud'] <= -37), 'RegionCluster'] = 3
    
    return df


def load_models():
    """Carga el preprocessor y el modelo"""
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        model = tf.keras.models.load_model('model.keras')
        return preprocessor, model
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}")
        sys.exit(1)


def main():
    print("="*70)
    print("PREDICCIÓN DE LLUVIA EN AUSTRALIA")
    print("="*70)
    
    # 1. Cargar input.csv
    try:
        print("\nCargando datos...")
        input_df = pd.read_csv('input.csv')
    except FileNotFoundError:
        print("Error: No se encontró el archivo input.csv")
        sys.exit(1)
    
    # 2. Verificar columnas requeridas
    required_cols = ['Location', 'Latitud', 'Longitud']
    missing_cols = [col for col in required_cols if col not in input_df.columns]
    
    if missing_cols:
        print(f"Error: Faltan columnas requeridas: {missing_cols}")
        sys.exit(1)
    
    # 3. Preparar datos
    if 'Date' in input_df.columns:
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df['Month'] = input_df['Date'].dt.month
    else:
        input_df['Month'] = pd.Timestamp.now().month
    
    # Asignar RegionCluster
    input_df = assign_region_cluster(input_df)
    
    # 4. Cargar modelos
    print("Cargando modelo...")
    preprocessor, model = load_models()
    
    # 5. Preprocesar
    print("Procesando...")
    output_info = input_df[['Location', 'Date']].copy() if 'Location' in input_df.columns and 'Date' in input_df.columns else pd.DataFrame()
    X = preprocessor.transform(input_df)
    
    # 6. Predecir
    predictions_proba = model.predict(X, verbose=0)
    predictions = (predictions_proba > 0.5).astype(int).flatten()
    
    # 7. Crear output
    output_df = pd.DataFrame()
    if not output_info.empty:
        output_df['Location'] = output_info['Location'].values
        output_df['Date'] = output_info['Date'].values
    
    output_df['Prediccion'] = ['Yes' if p == 1 else 'No' for p in predictions]
    output_df['Probabilidad_Lluvia'] = predictions_proba.flatten()
    output_df['Probabilidad_No_Lluvia'] = 1 - predictions_proba.flatten()
    
    # 8. Guardar output.csv
    output_df.to_csv('output.csv', index=False)
    
    # 9. Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS")
    print("="*70)
    
    for idx, row in output_df.iterrows():
        print(f"\nRegistro {idx + 1}:")
        if 'Location' in output_df.columns:
            print(f"  Ubicación: {row['Location']}")
        if 'Date' in output_df.columns:
            print(f"  Fecha: {row['Date']}")
        print(f"  ¿Lloverá mañana? {row['Prediccion']}")
        print(f"  Probabilidad: {row['Probabilidad_Lluvia']:.2%}")
    
    print("\n" + "="*70)
    print(f"Resultados guardados en output.csv")
    print("="*70)


if __name__ == "__main__":
    main()