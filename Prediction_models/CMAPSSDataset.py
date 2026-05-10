import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import torch

CONFIG_DATASETS = {
"FD001": {"n_condiciones": 1, "n_fallos": 1, "cnn_filtros": 64, "lstm_units": 64}, # Subset estable 
"FD002": {"n_condiciones": 6, "n_fallos": 1, "cnn_filtros": 64, "lstm_units": 128}, # Subset con más condiciones => más tipos de degradación temporal
"FD003": {"n_condiciones": 1, "n_fallos": 2, "cnn_filtros": 128, "lstm_units": 64}, # Más patrones posibles de degradación
"FD004": {"n_condiciones": 6, "n_fallos": 2, "cnn_filtros": 128, "lstm_units": 128} # Ambos
}

#Eliminar sensores constantes
def eliminar_constantes(df, umbral_std=0.01):
    sensor_cols = [c for c in df.columns if c.startswith("s")]
    std = df[sensor_cols].std()
    constantes = std[std < umbral_std].index.tolist()
    print(f"Sensores eliminados por baja varianza: {constantes}")
    return df.drop(columns=constantes), constantes

#Calcular y añadir la etiqueta RUL
def add_rul_train(df, rul_max=110):
    """
    RUL real = ciclos restantes hasta el fallo.
    Se trunca a rul_max (cap lineal): permite que el modelo se centre en los ciclos 
    cercanos al fallo. No hay necesidad de predecir el fallo desde el inicio del trayecto.
    """
    max_ciclo = df.groupby("motor_id")["ciclo"].max().rename("ciclo_max")
    df = df.join(max_ciclo, on="motor_id")
    df["rul"] = (df["ciclo_max"] - df["ciclo"]).clip(upper=rul_max)
    df = df.drop(columns=["ciclo_max"])
    return df

def add_rul_test(df, rul_finales, rul_max=110):
    """
    En test, NASA proporciona cuántos ciclos quedaban al final
    del fragmento observado. Se suman los ciclos restantes y se calcula
    el RUL para cada ciclo de cada motor
    """
    rul_map = {
        motor_id + 1: rul
        for motor_id, rul in enumerate(rul_finales["rul_final"].values)
    }
    max_ciclo = df.groupby("motor_id")["ciclo"].max()
    
    def rul_para_motor(row):
        rul_final = rul_map[row["motor_id"]]
        ciclos_restantes = (max_ciclo[row["motor_id"]] - row["ciclo"]) + rul_final
        return min(ciclos_restantes, rul_max)
    
    df["rul"] = df.apply(rul_para_motor, axis=1)
    return df

def norm(train_df, test_df, feature_cols):
    """
    MinMaxScaler ajustado SOLO sobre train.
    Aplica la misma transformación a test sin re-ajustar.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols]) 
    
    return train_df, test_df, scaler

def crear_ventanas(df, feature_cols, window_size=30):
    """
    Para cada motor, recorre sus ciclos con una ventana de tamaño window_size.
    Si el motor tiene menos ciclos que window_size, se rellena con padding
    al principio (ceros)
    
    Devuelve:
        X: (N, window_size, n_features)
        y: (N,)  — RUL en el último ciclo de cada ventana
    """
    X_list, y_list = [], []
    
    for motor_id, motor in df.groupby("motor_id"):
        datos = motor[feature_cols].values   # (ciclos, n_features)
        etiqueta = motor["rul"].values           # (ciclos,)
        n_ciclos = len(datos)
        
        if n_ciclos < window_size:
            # Padding al principio con ceros
            pad = np.zeros((window_size - n_ciclos, datos.shape[1]))
            datos = np.vstack([pad, datos])
            etiqueta = np.concatenate([np.full(window_size - n_ciclos, etiqueta[0]), etiqueta])
            n_ciclos = window_size
        
        for i in range(n_ciclos - window_size + 1):
            X_list.append(datos[i : i + window_size])
            y_list.append(etiqueta[i + window_size - 1])   # RUL del último ciclo
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def crear_ventanas_test(df, feature_cols, window_size=30):
    """
    Se usa la última ventana de cada motor
    """
    X_list, y_list = [], []
    
    for motor_id, motor in df.groupby("motor_id"):
        datos = motor[feature_cols].values
        etiqueta = motor["rul"].values
        n_ciclos = len(datos)
        
        if n_ciclos >= window_size:
            X_list.append(datos[-window_size:])
        else:
            pad = np.zeros((window_size - n_ciclos, datos.shape[1]))
            ventana = np.vstack([pad, datos])
            X_list.append(ventana)
        
        y_list.append(etiqueta[-1])
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def añadir_features_degradacion_rapido(df, sensor_cols, window_slope=10, suavizado=3):
    """
    Versión vectorizada — entre 20x y 50x más rápida que la versión con bucles.
    
    La clave es operar sobre columnas enteras en lugar de valor a valor:
    - delta:  una resta de arrays
    - slope:  rolling OLS vectorizado con fórmula analítica
    - accel:  diff sobre el slope suavizado
    """
    df = df.copy().sort_values(["motor_id", "ciclo"])
    result_chunks = []
    
    for motor_id, grupo in df.groupby("motor_id", sort=False):
        chunk = grupo.copy()
        n = len(chunk)
    
        for col in sensor_cols:
            serie = chunk[col].values  # array NumPy — operaciones vectorizadas
    
            # ── Delta: resta directa, sin bucle ───────────────────────────
            chunk[f"{col}_delta"] = serie - serie[0]
    
            # ── Slope: OLS analítico vectorizado ──────────────────────────
            # En una ventana [t-w+1 .. t], la pendiente OLS es:
            #   slope = (Σ(x-x̄)(y-ȳ)) / (Σ(x-x̄)²)
            # donde x son los índices 0..w-1, constantes para todas las ventanas
            # Esto se puede calcular con rolling sobre toda la columna de una vez
            s = pd.Series(serie)
    
            # Precomputar denominador — es constante para ventanas completas
            # x = [0, 1, ..., w-1], x̄ = (w-1)/2
            # Σ(x-x̄)² = w*(w²-1)/12  (fórmula cerrada)
            w = window_slope
            x_mean = (w - 1) / 2.0
            denom  = w * (w**2 - 1) / 12.0  # escalar, se calcula una vez
    
            # Numerador: rolling covarianza entre índices y valores
            # Σ(x-x̄)(y-ȳ) = Σ(x·y) - w·x̄·ȳ
            # Σ(x·y) con x=[0..w-1]: se puede expresar como suma ponderada
            weights_x = np.arange(w, dtype=np.float64) - x_mean  # (w,)
    
            # rolling_apply sobre toda la serie — una sola llamada
            def slope_ventana(y_ventana):
                if len(y_ventana) < 3:
                    return 0.0
                y_c = y_ventana - y_ventana.mean()
                return np.dot(weights_x[-len(y_ventana):], y_c) / denom
    
            slope = (s.rolling(window=w, min_periods=3)
                      .apply(slope_ventana, raw=True)
                      .fillna(0.0)
                      .values)
            chunk[f"{col}_slope"] = slope
    
            # ── Aceleración: diff del slope suavizado ─────────────────────
            slope_suav = (pd.Series(slope)
                           .rolling(window=suavizado, min_periods=1)
                           .mean()
                           .values)
            chunk[f"{col}_accel"] = np.gradient(slope_suav)
    
        result_chunks.append(chunk)
    
    return pd.concat(result_chunks, ignore_index=True)

def identificar_condicion_operacional(df, n_condiciones=6):
    """
    En FD002/FD004 hay 6 condiciones operacionales distintas.
    Las identificamos agrupando op_1, op_2, op_3 con KMeans.
    En FD001/FD003 con n_condiciones=1 asigna todo al cluster 0.
    """
    op_cols = ["op_1", "op_2", "op_3"]
    
    if n_condiciones == 1:
        df["condicion"] = 0
        return df
    
    km = KMeans(n_clusters=n_condiciones, random_state=42, n_init=10)
    df["condicion"] = km.fit_predict(df[op_cols])
    return df, km


def normalizar_por_condicion(train_df, test_df, feature_cols, n_condiciones):
    """
    Para FD002/FD004: ajusta un scaler distinto por condición operacional.
    Para FD001/FD003: normalización global (igual que antes).
    
    Esto evita que el modelo confunda variación por régimen de vuelo
    con variación por degradación.
    """
    train_df[feature_cols] = train_df[feature_cols].astype(np.float64)
    test_df[feature_cols]  = test_df[feature_cols].astype(np.float64)
    
    if n_condiciones == 1:
        scaler = MinMaxScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
        return train_df, test_df, {0: scaler}
    
    scalers = {}
    for cond in range(n_condiciones):
        mask_train = train_df["condicion"] == cond
        mask_test = test_df["condicion"]  == cond
        
        if mask_train.sum() == 0:
            continue
            
        scaler = MinMaxScaler()
        train_df.loc[mask_train, feature_cols] = scaler.fit_transform(
            train_df.loc[mask_train, feature_cols]
        )
        if mask_test.sum() > 0:
            test_df.loc[mask_test, feature_cols] = scaler.transform(
                test_df.loc[mask_test, feature_cols]
            )
        scalers[cond] = scaler
    
    return train_df, test_df, scalers

def extraer_condicion_por_ventana(df, window_size, n_condiciones):
    """
    Para cada ventana de entrenamiento extrae la condición operacional
    predominante y la convierte en one-hot.

    Salida: (n_ventanas, n_condiciones) como numpy float32
    """
    static_list = []

    for motor_id, grupo in df.groupby("motor_id"):
        condiciones = grupo["condicion"].values
        n_ciclos    = len(condiciones)

        if n_ciclos < window_size:
            # ventana con padding — usar la condición del primer ciclo
            n_ventanas = 1
            cond_predominante = [int(condiciones[0])]
        else:
            n_ventanas = n_ciclos - window_size + 1
            cond_predominante = []
            for i in range(n_ventanas):
                ventana_cond = condiciones[i:i + window_size]
                # condición más frecuente en la ventana
                cond = int(np.bincount(ventana_cond).argmax())
                cond_predominante.append(cond)

        # one-hot encoding
        for cond in cond_predominante:
            onehot = np.zeros(n_condiciones, dtype=np.float32)
            onehot[cond] = 1.0
            static_list.append(onehot)

    return np.array(static_list, dtype=np.float32)


def extraer_condicion_por_ventana_test(df, window_size, n_condiciones):
    """
    Para test solo tomamos la última ventana de cada motor.
    """
    static_list = []

    for motor_id, grupo in df.groupby("motor_id"):
        condiciones = grupo["condicion"].values
        n_ciclos    = len(condiciones)

        if n_ciclos >= window_size:
            ventana_cond = condiciones[-window_size:]
        else:
            ventana_cond = condiciones

        cond = int(np.bincount(ventana_cond).argmax())
        onehot = np.zeros(n_condiciones, dtype=np.float32)
        onehot[cond] = 1.0
        static_list.append(onehot)

    return np.array(static_list, dtype=np.float32)

class CMAPSSDataset():
    def __init__(self, X, y, static=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.static = torch.tensor(static, dtype=torch.float32) if static is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.static is not None:
            return self.X[idx], self.y[idx], self.static[idx]
        return self.X[idx], self.y[idx]