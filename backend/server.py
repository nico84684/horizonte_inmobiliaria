from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Constantes ---
OUTPUT_FILE_NAME = "PropiedadesLimpio_v4.csv"
SCRIPT_NAME = "clean_data_v4.py"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, OUTPUT_FILE_NAME)
SCRIPT_PATH = os.path.join(BASE_DIR, SCRIPT_NAME)
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
TRAINING_UPLOAD = os.path.join(UPLOADS_DIR, "entrenamiento.csv")
DOLAR_UPLOAD = os.path.join(UPLOADS_DIR, "DOLAR OFICIAL - Cotizaciones historicas.csv")
PRICING_DATASET_NAME = "pricing_dataset.csv"
PRICING_DATASET_PATH = os.path.join(BASE_DIR, PRICING_DATASET_NAME)
PRICING_MODEL_PATH = os.path.join(BASE_DIR, "pricing_model.pkl")
MI_DAN_PATH = os.path.join(BASE_DIR, "..", "Datasets crudos", "MI_DAN_AX03.xlsx")
MONTH_MAP = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


@app.route("/api/last-execution", methods=["GET"])
def get_last_execution_time():
    """Endpoint para obtener la fecha de última modificación del archivo de salida."""
    if not os.path.exists(OUTPUT_FILE_PATH):
        return jsonify(last_execution_date=None, file_exists=False)

    try:
        last_mod_time = os.path.getmtime(OUTPUT_FILE_PATH)
        last_execution_date = datetime.fromtimestamp(last_mod_time).strftime("%d/%m/%Y, %H:%M:%S")
        return jsonify(last_execution_date=last_execution_date, file_exists=True)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/api/sources", methods=["GET"])
def get_sources():
    """Devuelve qué archivos fuente están configurados actualmente."""
    return jsonify({
        "training_path": TRAINING_UPLOAD if os.path.exists(TRAINING_UPLOAD) else None,
        "dolar_path": DOLAR_UPLOAD if os.path.exists(DOLAR_UPLOAD) else None,
        "defaults": True  # indicador simple por ahora
    })


@app.route("/api/pricing-dataset", methods=["GET"])
def get_pricing_dataset_info():
    """Devuelve información del dataset listo para pricing si ya existe."""
    exists = os.path.exists(PRICING_DATASET_PATH)
    last_updated = None
    rows = None

    if exists:
        try:
            last_mod_time = os.path.getmtime(PRICING_DATASET_PATH)
            last_updated = datetime.fromtimestamp(last_mod_time).strftime("%d/%m/%Y, %H:%M:%S")
            # Conteo ligero de filas sin cargar todo en memoria
            with open(PRICING_DATASET_PATH, "r", encoding="utf-8", errors="ignore") as file:
                rows = max(sum(1 for _ in file) - 1, 0)
        except Exception as exc:  # pragma: no cover - evita romper la UI
            return jsonify(error=str(exc)), 500

    return jsonify({
        "exists": exists,
        "last_updated": last_updated,
        "rows": rows,
        "path": PRICING_DATASET_PATH,
    })


@app.route("/api/train-pricing-model", methods=["POST"])
def train_pricing_model():
    """Entrena el modelo de pricing y guarda el pipeline serializado."""
    try:
        metrics = _train_pricing_model()
        global _model_cache
        _model_cache = joblib.load(PRICING_MODEL_PATH)
        _model_cache["metrics"] = metrics
        return jsonify({"message": "Modelo entrenado", "metrics": metrics, "model_path": PRICING_MODEL_PATH})
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.route("/api/price-predict", methods=["POST"])
def price_predict():
    """Predice el precio recomendado para una propiedad."""
    payload = request.get_json(force=True, silent=True) or {}
    try:
        model_bundle = _get_or_train_model()
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        return jsonify(error=f"No se pudo cargar el modelo: {exc}"), 500

    required_fields = [
        "tipo_propiedad",
        "tipo_operacion",
        "ambientes",
        "banios",
        "superficie_total",
        "superficie_cubierta",
        "latitud",
        "longitud",
        "partido",
        "provincia",
    ]
    missing = [f for f in required_fields if f not in payload]
    if missing:
        return jsonify(error=f"Faltan campos: {', '.join(missing)}"), 400

    df_row = pd.DataFrame([payload])
    # Derivar ratio cubierta
    try:
        df_row["ratio_cubierta"] = df_row["superficie_cubierta"] / df_row["superficie_total"]
    except Exception:
        df_row["ratio_cubierta"] = pd.NA
    # Valor opcional de precio publicado (solo para calcular delta)
    price_publish = payload.get("precio_publicado")
    # Feature opcional: precio_m2_publicado, se deja como NaN para que el imputador lo maneje
    df_row["precio_m2_publicado"] = np.nan
    # Derivar fecha y mi_dan si no viene
    fecha_str = payload.get("fecha_publicacion") or payload.get("fecha_creacion")
    if fecha_str:
        df_row["fecha_creacion"] = pd.to_datetime(fecha_str, errors="coerce")
    else:
        df_row["fecha_creacion"] = pd.NaT

    if "mi_dan_ax03" not in df_row.columns or pd.isna(df_row.at[0, "mi_dan_ax03"]):
        mi_dan_df = _load_mi_dan_index()
        if mi_dan_df is not None and pd.notna(df_row.at[0, "fecha_creacion"]):
            df_row["periodo_publicacion"] = df_row["fecha_creacion"].dt.to_period("M")
            df_row = df_row.merge(mi_dan_df, how="left", left_on="periodo_publicacion", right_on="periodo")
            df_row["mi_dan_ax03"] = df_row.apply(_select_mi_dan, axis=1)
            df_row.drop(columns=["periodo", "mi_dan_1amb", "mi_dan_2amb", "mi_dan_3amb", "mi_dan_promedio"], inplace=True, errors="ignore")
    else:
        df_row["mi_dan_ax03"] = pd.NA

    # Precio publicado opcional para delta
    price_publish = payload.get("precio_publicado")

    # Asegurar columnas numéricas presentes y en float
    numeric_cols = [
        "latitud",
        "longitud",
        "ambientes",
        "dormitorios",
        "banios",
        "superficie_total",
        "superficie_cubierta",
        "ratio_cubierta",
        "precio_m2_publicado",
        "mi_dan_ax03",
    ]
    for col in numeric_cols:
        if col not in df_row.columns:
            df_row[col] = np.nan
        df_row[col] = pd.to_numeric(df_row[col], errors="coerce")

    op_key = str(payload.get("tipo_operacion", "")).strip().lower() or "venta"
    pipelines = model_bundle.get("pipelines") or {}
    pipeline = pipelines.get(op_key) or next(iter(pipelines.values()), None)
    if pipeline is None:
        return jsonify(error="No hay un modelo entrenado disponible."), 500

    prediction = float(pipeline.predict(df_row)[0])
    band_pct = 0.08
    price_min = round(prediction * (1 - band_pct), 2)
    price_max = round(prediction * (1 + band_pct), 2)
    delta = None
    if price_publish is not None:
        try:
            price_publish = float(price_publish)
            delta = round((price_publish - prediction) / prediction * 100, 2)
        except Exception:
            delta = None

    return jsonify({
        "predicted_price": round(prediction, 2),
        "price_min": price_min,
        "price_max": price_max,
        "delta_vs_publicado_pct": delta,
    })

@app.route("/api/upload-source", methods=["POST"])
def upload_source():
    """Permite subir los CSV de entrenamiento y cotización dolar."""
    if "training" not in request.files and "dolar" not in request.files:
        return jsonify(error="Envía al menos un archivo con claves 'training' o 'dolar'"), 400

    saved = {}
    for key, dest in [("training", TRAINING_UPLOAD), ("dolar", DOLAR_UPLOAD)]:
        file = request.files.get(key)
        if file:
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(".csv"):
                return jsonify(error=f"El archivo de {key} debe ser CSV"), 400
            file.save(dest)
            saved[key] = dest

    return jsonify(message="Archivos cargados correctamente", saved=saved)


def _emit_pipeline_finished(action: str):
    """Envía el evento de finalización con la fecha del archivo si existe."""
    last_execution_date = None
    if os.path.exists(OUTPUT_FILE_PATH):
        last_mod_time = os.path.getmtime(OUTPUT_FILE_PATH)
        last_execution_date = datetime.fromtimestamp(last_mod_time).strftime("%d/%m/%Y, %H:%M:%S")

    socketio.emit(
        "pipeline_finished",
        {
            "message": f"Proceso {action} completado.",
            "last_execution_date": last_execution_date,
            "action": action,
        },
    )


def _load_mi_dan_index() -> Optional[pd.DataFrame]:
    """Carga el índice MI_DAN_AX03 y lo prepara por mes para merge rápido."""
    if not os.path.exists(MI_DAN_PATH):
        return None

    raw = pd.read_excel(MI_DAN_PATH, header=None)
    data = raw.iloc[2:].copy()
    data.rename(columns={
        0: "anio",
        1: "mes",
        2: "mi_dan_promedio",
        3: "mi_dan_1amb",
        4: "mi_dan_2amb",
        5: "mi_dan_3amb",
    }, inplace=True)

    data["anio"] = data["anio"].ffill()
    data["mes"] = data["mes"].ffill()
    data["anio"] = pd.to_numeric(data["anio"], errors="coerce")
    data["mes"] = data["mes"].astype(str).str.strip().str.lower()
    data.dropna(subset=["anio", "mes"], inplace=True)
    data["mes_num"] = data["mes"].map(MONTH_MAP)
    data.dropna(subset=["mes_num"], inplace=True)
    data["fecha_indice"] = pd.to_datetime({
        "year": data["anio"],
        "month": data["mes_num"],
        "day": 1,
    })
    data["periodo"] = data["fecha_indice"].dt.to_period("M")

    return data[["periodo", "mi_dan_promedio", "mi_dan_1amb", "mi_dan_2amb", "mi_dan_3amb"]]


def _select_mi_dan(row: pd.Series) -> float | None:
    """Devuelve el índice MI_DAN según ambientes; fallback al promedio."""
    ambientes = row.get("ambientes")
    if pd.isna(ambientes):
        return row.get("mi_dan_promedio")
    try:
        ambientes_val = float(ambientes)
    except (TypeError, ValueError):
        return row.get("mi_dan_promedio")

    if ambientes_val <= 1:
        return row.get("mi_dan_1amb")
    if ambientes_val == 2:
        return row.get("mi_dan_2amb")
    return row.get("mi_dan_3amb")


def build_pricing_dataset():
    """Genera un dataset con features para pricing (modelo de precios)."""
    step = "pricing"
    socketio.emit("status", {"message": "Iniciando armado de dataset de pricing...", "action": step})

    if not os.path.exists(OUTPUT_FILE_PATH):
        socketio.emit("status", {
            "message": "No existe el archivo limpio. Ejecuta el pipeline completo antes de crear el dataset de pricing.",
            "action": step,
        })
        return

    try:
        df = pd.read_csv(OUTPUT_FILE_PATH, low_memory=False, encoding="utf-8")
        socketio.emit("status", {
            "message": f"Archivo base cargado ({len(df)} registros). Calculando features...",
            "action": step,
        })

        mi_dan_df = _load_mi_dan_index()
        if mi_dan_df is None:
            socketio.emit("status", {
                "message": "No se encontró MI_DAN_AX03.xlsx. Se generará el dataset sin este índice.",
                "action": step,
            })

        df["fecha_creacion_dt"] = pd.to_datetime(df["fecha_creacion"], errors="coerce")
        df["periodo_publicacion"] = df["fecha_creacion_dt"].dt.to_period("M")
        df["precio_m2_publicado"] = df["precio_dolares"] / df["superficie_total"]
        df["precio_m2_publicado"].replace([float("inf"), float("-inf")], pd.NA, inplace=True)

        if mi_dan_df is not None:
            df = df.merge(mi_dan_df, how="left", left_on="periodo_publicacion", right_on="periodo")
            df["mi_dan_ax03"] = df.apply(_select_mi_dan, axis=1)
            df.drop(columns=["periodo", "mi_dan_1amb", "mi_dan_2amb", "mi_dan_3amb", "mi_dan_promedio"], inplace=True, errors="ignore")
        else:
            df["mi_dan_ax03"] = pd.NA

        columnas_modelo = [
            "id",
            "latitud",
            "longitud",
            "ambientes",
            "dormitorios",
            "banios",
            "superficie_total",
            "superficie_cubierta",
            "tipo_propiedad",
            "tipo_operacion",
            "pais",
            "provincia",
            "partido",
            "localidad",
            "precio_dolares",
            "precio_m2_publicado",
            "mi_dan_ax03",
            "fecha_creacion_dt",
        ]
        df_modelo = df[[col for col in columnas_modelo if col in df.columns]].copy()
        df_modelo.rename(columns={"fecha_creacion_dt": "fecha_creacion"}, inplace=True)
        df_modelo.to_csv(PRICING_DATASET_PATH, index=False, encoding="utf-8-sig")

        socketio.emit("status", {
            "message": f"Dataset de pricing generado en '{PRICING_DATASET_PATH}' con {len(df_modelo)} filas.",
            "action": step,
        })
        socketio.emit("pricing_dataset_ready", {
            "path": PRICING_DATASET_PATH,
            "rows": len(df_modelo),
            "last_updated": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        })
        socketio.emit("pipeline_finished", {
            "message": "Dataset de pricing generado.",
            "last_execution_date": None,
            "action": step,
        })
    except Exception as exc:
        socketio.emit("status", {
            "message": f"Error al construir el dataset de pricing: {exc}",
            "action": step,
        })
        socketio.emit("pipeline_finished", {
            "message": "Error al generar dataset de pricing.",
            "last_execution_date": None,
            "action": step,
        })


def run_step(step: str):
    """Ejecuta el script con la etapa solicitada: clean, convert o all."""
    socketio.emit("status", {"message": f"Iniciando etapa: {step}..."})
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Si existen uploads, indicarlos al script
        if os.path.exists(TRAINING_UPLOAD):
            env["TRAINING_CSV_PATH"] = TRAINING_UPLOAD
        if os.path.exists(DOLAR_UPLOAD):
            env["DOLAR_CSV_PATH"] = DOLAR_UPLOAD

        args = [sys.executable, "-u", SCRIPT_PATH]
        if step != "all":
            args.extend(["--step", step])

        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            cwd=BASE_DIR,
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            clean_line = line.strip()
            if clean_line:
                print(f"Output: {clean_line}")
                socketio.emit("status", {"message": clean_line, "action": step})

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            socketio.emit("status", {"message": f"Etapa {step} finalizada con éxito.", "action": step})
            _emit_pipeline_finished(step)
        else:
            socketio.emit("status", {"message": f"Error en la etapa {step}. Código de salida: {return_code}", "action": step})

    except FileNotFoundError:
        error_msg = f"Error: El script '{SCRIPT_NAME}' no se encontró en la ruta esperada."
        print(error_msg)
        socketio.emit("status", {"message": error_msg, "action": step})
    except Exception as e:
        error_msg = f"Ocurrió un error inesperado al ejecutar la etapa {step}: {str(e)}"
        print(error_msg)
        socketio.emit("status", {"message": error_msg, "action": step})


@socketio.on("connect")
def handle_connect():
    """Confirma la conexión del cliente."""
    print("Cliente conectado")
    socketio.emit("status", {"message": "Conectado al servidor."})


@socketio.on("disconnect")
def handle_disconnect():
    """Maneja la desconexión del cliente."""
    print("Cliente desconectado")


@socketio.on("run_pipeline")
def handle_run_pipeline():
    """Compatibilidad: ejecuta el pipeline completo."""
    run_step("all")


@socketio.on("run_full")
def handle_run_full():
    """Ejecuta el pipeline completo."""
    run_step("all")


@socketio.on("run_clean")
def handle_run_clean():
    """Ejecuta solo la etapa de limpieza."""
    run_step("clean")


@socketio.on("run_convert")
def handle_run_convert():
    """Ejecuta solo la etapa de conversión."""
    run_step("convert")


@socketio.on("run_pricing_dataset")
def handle_run_pricing_dataset():
    """Genera el dataset listo para el modelo de precios."""
    build_pricing_dataset()


def _train_pricing_model():
    """Entrena modelos Gradient Boosting segmentados por operación sobre el dataset de pricing."""
    if not os.path.exists(PRICING_DATASET_PATH):
        raise FileNotFoundError(f"No se encontró {PRICING_DATASET_PATH}. Genera el dataset de pricing primero.")

    df = pd.read_csv(PRICING_DATASET_PATH, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()]  # evitar columnas repetidas
    if "precio_dolares" not in df.columns:
        raise ValueError("El dataset de pricing no contiene la columna objetivo 'precio_dolares'.")

    target = "precio_dolares"
    feature_cols = [
        "latitud",
        "longitud",
        "ambientes",
        "dormitorios",
        "banios",
        "superficie_total",
        "superficie_cubierta",
        "ratio_cubierta",
        "precio_m2_publicado",
        "mi_dan_ax03",
        "tipo_propiedad",
        "tipo_operacion",
        "provincia",
        "partido",
    ]
    selected_cols = list(dict.fromkeys(feature_cols + [target, "tipo_operacion"]))
    df = df[[col for col in selected_cols if col in df.columns]].copy()
    df["ratio_cubierta"] = df["superficie_cubierta"] / df["superficie_total"]
    if "precio_m2_publicado" not in df.columns or df["precio_m2_publicado"].isna().all():
        df["precio_m2_publicado"] = df[target] / df["superficie_total"]
    df = df.dropna(subset=[target])

    # Filtrado de precios para evitar valores no razonables
    prices = df[target].to_numpy()
    ops_series = df["tipo_operacion"] if "tipo_operacion" in df.columns else pd.Series([], dtype=str)
    if isinstance(ops_series, pd.DataFrame):
        ops_series = ops_series.iloc[:, 0]
    ops = ops_series.astype(str).str.lower().to_numpy()
    mask_price = prices > 0
    mask_venta = (ops != "venta") | ((prices >= 20_000) & (prices <= 2_000_000))
    mask_alquiler = (ops != "alquiler") | ((prices >= 100) & (prices <= 10_000))
    mask = mask_price & mask_venta & mask_alquiler
    df = df.loc[mask].copy()

    # Filtrado de precio/m2 por segmento (operación + tipo_propiedad)
    if "tipo_propiedad" in df.columns:
        df["precio_m2_publicado"] = df["precio_m2_publicado"].replace([float("inf"), float("-inf")], pd.NA)
        df.dropna(subset=["precio_m2_publicado"], inplace=True)
        filtered_frames = []
        for (op, tip), g in df.groupby(["tipo_operacion", "tipo_propiedad"]):
            if len(g) < 50:
                filtered_frames.append(g)
                continue
            low, high = g["precio_m2_publicado"].quantile([0.01, 0.99])
            filtered_frames.append(g[(g["precio_m2_publicado"] >= low) & (g["precio_m2_publicado"] <= high)])
        if filtered_frames:
            df = pd.concat(filtered_frames, ignore_index=True)

    numeric_cols = [
        "latitud",
        "longitud",
        "ambientes",
        "dormitorios",
        "banios",
        "superficie_total",
        "superficie_cubierta",
        "ratio_cubierta",
        "precio_m2_publicado",
        "mi_dan_ax03",
    ]
    categorical_cols = [
        "tipo_propiedad",
        "tipo_operacion",
        "provincia",
        "partido",
    ]

    def _safe_mape(y_true, preds):
        try:
            y_true = pd.Series(y_true)
            preds = pd.Series(preds, index=y_true.index)
            mask = y_true >= 10_000
            if mask.any():
                return float(mean_absolute_percentage_error(y_true[mask], preds[mask]) * 100)
            return None
        except Exception:
            return None

    pipelines = {}
    metrics = {"overall": {}, "by_operation": {}, "by_tipo_propiedad": {}}
    all_true = []
    all_pred = []

    for op in df["tipo_operacion"].dropna().unique():
        subset = df[df["tipo_operacion"] == op]
        X = subset[feature_cols]
        y = subset[target]

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, [c for c in numeric_cols if c in X.columns]),
                ("cat", categorical_transformer, [c for c in categorical_cols if c in X.columns]),
            ]
        )
        model = GradientBoostingRegressor(random_state=42)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        pipelines[op.lower()] = pipeline

        mae_op = float(mean_absolute_error(y_val, preds))
        mape_op = _safe_mape(y_val, preds)
        metrics["by_operation"][op] = {"mae": mae_op, "mape_pct": mape_op}

        # métricas por tipo_propiedad dentro de la operación
        metrics_prop = {}
        for prop in subset["tipo_propiedad"].dropna().unique():
            mask_prop = X_val["tipo_propiedad"] == prop
            if mask_prop.any():
                mae_p = float(mean_absolute_error(y_val[mask_prop], preds[mask_prop]))
                mape_p = _safe_mape(y_val[mask_prop], preds[mask_prop])
                metrics_prop[prop] = {"mae": mae_p, "mape_pct": mape_p}
        metrics["by_tipo_propiedad"][op] = metrics_prop

        all_true.append(y_val)
        all_pred.append(pd.Series(preds, index=y_val.index))

    if all_true:
        y_all = pd.concat(all_true)
        p_all = pd.concat(all_pred)
        metrics["overall"]["mae"] = float(mean_absolute_error(y_all, p_all))
        metrics["overall"]["mape_pct"] = _safe_mape(y_all, p_all)

    bundle = {
        "pipelines": pipelines,
        "trained_at": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
        "metrics": metrics,
    }
    joblib.dump(bundle, PRICING_MODEL_PATH)
    return metrics


_model_cache: dict | None = None


def _get_or_train_model():
    """Carga el modelo desde disco o lo entrena si no existe."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if os.path.exists(PRICING_MODEL_PATH):
        _model_cache = joblib.load(PRICING_MODEL_PATH)
        return _model_cache
    metrics = _train_pricing_model()
    _model_cache = joblib.load(PRICING_MODEL_PATH)
    _model_cache["metrics"] = metrics
    return _model_cache

if __name__ == "__main__":
    print("Iniciando servidor Flask...")
    socketio.run(app, debug=True, port=5000, host="0.0.0.0")
