import { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';
import './App.css';
import logo from '/logoHorizonte.png';

const BACKEND_URL = 'http://localhost:5000';
const socket = io(BACKEND_URL);

type ActionType = 'full' | 'clean' | 'convert' | 'pricing';
type TabKey = 'pipeline' | 'pricing' | 'dashboard' | 'chatbot' | 'docs';
type TrainingMetrics = { mae?: number | null; mape_pct?: number | null };

const PROPERTY_TYPES = ['Departamento', 'Casa', 'PH', 'Oficina', 'Local', 'Terreno'];
const OPERATIONS = ['Venta', 'Alquiler'];
const PROVINCIAS = ['Buenos Aires', 'CABA'];
const PARTIDOS = ['CABA', 'Vicente López', 'San Isidro', 'Tigre', 'La Plata'];

const ACTION_LABELS: Record<ActionType, string> = {
  full: 'Pipeline completo',
  clean: 'Limpieza',
  convert: 'Conversión',
  pricing: 'Dataset de pricing',
};

const ACTION_EVENT: Record<ActionType, string> = {
  full: 'run_full',
  clean: 'run_clean',
  convert: 'run_convert',
  pricing: 'run_pricing_dataset',
};

function App() {
  const [lastExecutionDate, setLastExecutionDate] = useState<string | null>('Cargando...');
  const [statusMessages, setStatusMessages] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [currentAction, setCurrentAction] = useState<ActionType | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>('pipeline');
  const [trainingFile, setTrainingFile] = useState<File | null>(null);
  const [dolarFile, setDolarFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [uploading, setUploading] = useState<boolean>(false);
  const [sources, setSources] = useState<{ training_path?: string | null; dolar_path?: string | null }>({});
  const [pricingDataset, setPricingDataset] = useState<{ last_updated: string | null; rows: number | null; path: string | null }>({
    last_updated: null,
    rows: null,
    path: null,
  });
  const [trainingModel, setTrainingModel] = useState<{ loading: boolean; message: string; metrics?: TrainingMetrics }>({
    loading: false,
    message: '',
    metrics: undefined,
  });
  const [predictForm, setPredictForm] = useState({
    tipo_propiedad: 'Departamento',
    tipo_operacion: 'Venta',
    ambientes: '2',
    dormitorios: '1',
    banios: '1',
    superficie_total: '50',
    superficie_cubierta: '45',
    latitud: '',
    longitud: '',
    partido: '',
    provincia: '',
    localidad: '',
    fecha_publicacion: '',
    precio_publicado: '',
  });
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult] = useState<{ predicted_price: number; price_min: number; price_max: number; delta_vs_publicado_pct?: number | null } | null>(null);
  const statusLogRef = useRef<HTMLDivElement>(null);
  const dashboardFrameRef = useRef<HTMLIFrameElement | null>(null);
  const chatbotFrameRef = useRef<HTMLIFrameElement | null>(null);

  const businessMessage = (raw: string) => {
    const lower = raw.toLowerCase();
    const mappings: { match: string; text: string }[] = [
      { match: 'iniciando', text: 'Arrancando el proceso de depuración y control de datos.' },
      { match: 'cargando cotizaciones', text: 'Tomando la referencia del dólar para convertir precios.' },
      { match: 'eliminando columnas', text: 'Depurando campos innecesarios de la base.' },
      { match: 'intercambiando latitud', text: 'Reordenando ubicaciones geográficas.' },
      { match: 'renombrando columnas', text: 'Normalizando nombres para análisis.' },
      { match: 'filtrando monedas', text: 'Quitando publicaciones en moneda no soportada.' },
      { match: 'convirtiendo precios', text: 'Unificando precios en dólares.' },
      { match: 'eliminando filas con datos faltantes', text: 'Conservando solo avisos con datos completos.' },
      { match: 'filtrando outliers', text: 'Descartando valores fuera de mercado.' },
      { match: 'guardando', text: 'Guardando el archivo final para el panel.' },
      { match: 'pipeline finalizado', text: 'Pipeline listo. Datos actualizados.' },
      { match: 'etapa clean finalizada', text: 'Limpieza lista. Datos depurados.' },
      { match: 'etapa convert finalizada', text: 'Conversión lista. Precios unificados.' },
      { match: 'armado de dataset de pricing', text: 'Preparando el dataset para el modelo de precios.' },
      { match: 'archivo base cargado', text: 'Usando el dataset limpio para derivar variables de pricing.' },
      { match: 'mi_dan', text: 'Integrando el índice MI_DAN como referencia de m².' },
      { match: 'dataset de pricing', text: 'Dataset para pricing generado con features listas.' },
      { match: 'no se encontró mi_dan', text: 'No se halló MI_DAN_AX03.xlsx. Se generará sin ese índice.' },
      { match: 'error', text: 'Se detectó un problema. Revisar los datos de origen.' },
    ];

    const found = mappings.find(item => lower.includes(item.match));
    return found ? found.text : raw;
  };

  useEffect(() => {
    const fetchLastExecution = () => {
      fetch(`${BACKEND_URL}/api/last-execution`)
        .then(res => res.json())
        .then(data => {
          if (data.file_exists) {
            setLastExecutionDate(data.last_execution_date);
          } else {
            setLastExecutionDate('No se ha ejecutado aún');
          }
        })
        .catch(() => {
          setLastExecutionDate('Error al contactar al servidor');
        });
    };

    const fetchSources = () => {
      fetch(`${BACKEND_URL}/api/sources`)
        .then(res => res.json())
        .then(data => setSources(data))
        .catch(() => setSources({}));
    };

    const fetchPricingDataset = () => {
      fetch(`${BACKEND_URL}/api/pricing-dataset`)
        .then(res => res.json())
        .then(data => {
          setPricingDataset({
            last_updated: data.last_updated ?? null,
            rows: data.rows ?? null,
            path: data.path ?? null,
          });
        })
        .catch(() => {
          setPricingDataset({
            last_updated: null,
            rows: null,
            path: null,
          });
        });
    };

    fetchLastExecution();
    fetchSources();
    fetchPricingDataset();

    const onConnect = () => {
      console.log('Conectado al servidor de Socket.IO');
    };

    const onPricingDatasetReady = (data: { path?: string; rows?: number; last_updated?: string | null }) => {
      setPricingDataset({
        last_updated: data.last_updated ?? new Date().toLocaleString('es-AR'),
        rows: data.rows ?? null,
        path: data.path ?? null,
      });
    };

    const onStatusMessage = (data: { message: string; action?: ActionType }) => {
      const friendly = businessMessage(data.message);
      setStatusMessages(prevMessages => [...prevMessages, friendly]);
      setProgress(prev => Math.min(90, prev + 8));
    };

    const onPipelineFinished = (data: { last_execution_date: string | null; action?: ActionType }) => {
      if (data.last_execution_date) {
        setLastExecutionDate(data.last_execution_date);
      }
      setIsRunning(false);
      setCurrentAction(null);
      setProgress(100);
      setStatusMessages(prev => [...prev, 'Proceso finalizado. Archivo actualizado y listo para el negocio.']);
    };

    socket.on('connect', onConnect);
    socket.on('status', onStatusMessage);
    socket.on('pipeline_finished', onPipelineFinished);
    socket.on('pricing_dataset_ready', onPricingDatasetReady);

    return () => {
      socket.off('connect', onConnect);
      socket.off('status', onStatusMessage);
      socket.off('pipeline_finished', onPipelineFinished);
      socket.off('pricing_dataset_ready', onPricingDatasetReady);
    };
  }, []);

  useEffect(() => {
    if (statusLogRef.current) {
      statusLogRef.current.scrollTop = statusLogRef.current.scrollHeight;
    }
  }, [statusMessages]);

  const handlePredictChange = (field: string, value: string) => {
    setPredictForm(prev => ({ ...prev, [field]: value }));
  };

  const scrollToDoc = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handlePredict = () => {
    setPredictLoading(true);
    setPredictResult(null);

    const payload: Record<string, any> = {
      tipo_propiedad: predictForm.tipo_propiedad,
      tipo_operacion: predictForm.tipo_operacion,
      ambientes: predictForm.ambientes ? Number(predictForm.ambientes) : null,
      dormitorios: predictForm.dormitorios ? Number(predictForm.dormitorios) : null,
      banios: predictForm.banios ? Number(predictForm.banios) : null,
      superficie_total: predictForm.superficie_total ? Number(predictForm.superficie_total) : null,
      superficie_cubierta: predictForm.superficie_cubierta ? Number(predictForm.superficie_cubierta) : null,
      latitud: predictForm.latitud ? Number(predictForm.latitud) : null,
      longitud: predictForm.longitud ? Number(predictForm.longitud) : null,
      partido: predictForm.partido,
      provincia: predictForm.provincia,
      localidad: predictForm.localidad,
      fecha_publicacion: predictForm.fecha_publicacion,
    };

    if (predictForm.precio_publicado) {
      payload.precio_publicado = Number(predictForm.precio_publicado);
    }

    fetch(`${BACKEND_URL}/api/price-predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setPredictResult(null);
          setStatusMessages(prev => [...prev, data.error]);
        } else {
          setPredictResult(data);
        }
      })
      .catch(() => {
        setPredictResult(null);
        setStatusMessages(prev => [...prev, 'No se pudo obtener la predicción.']);
      })
      .finally(() => setPredictLoading(false));
  };

  const handleDashboardFullscreen = () => {
    const iframe = dashboardFrameRef.current;
    if (!iframe) return;
    const elem: any = iframe;
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) {
      elem.msRequestFullscreen();
    }
  };

  const handleChatbotFullscreen = () => {
    const iframe = chatbotFrameRef.current;
    if (!iframe) return;
    const elem: any = iframe;
    if (elem.requestFullscreen) {
      elem.requestFullscreen();
    } else if (elem.webkitRequestFullscreen) {
      elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) {
      elem.msRequestFullscreen();
    }
  };

  const handleRetrainModel = () => {
    setTrainingModel({ loading: true, message: 'Entrenando modelo...' });
    fetch(`${BACKEND_URL}/api/train-pricing-model`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setTrainingModel({ loading: false, message: data.error, metrics: undefined });
          setStatusMessages(prev => [...prev, data.error]);
        } else {
          const mae = data.metrics?.overall?.mae ?? data.metrics?.mae;
          const mape = data.metrics?.overall?.mape_pct ?? data.metrics?.mape_pct;
          const maeTxt = typeof mae === 'number' ? mae.toFixed(0) : '-';
          const mapeTxt = typeof mape === 'number' ? `${mape.toFixed(2)}%` : '-';
          const msg = `Modelo entrenado. MAE: ${maeTxt} | MAPE: ${mapeTxt}`;
          setTrainingModel({
            loading: false,
            message: msg,
            metrics: { mae: typeof mae === 'number' ? mae : null, mape_pct: typeof mape === 'number' ? mape : null },
          });
          setStatusMessages(prev => [...prev, msg]);
        }
      })
      .catch(() => {
        setTrainingModel({ loading: false, message: 'No se pudo entrenar el modelo.', metrics: undefined });
        setStatusMessages(prev => [...prev, 'No se pudo entrenar el modelo.']);
      });
  };

  const handleUploadSources = () => {
    if (!trainingFile && !dolarFile) {
      setUploadStatus('Selecciona al menos un archivo para cargar.');
      return;
    }
    const formData = new FormData();
    if (trainingFile) formData.append('training', trainingFile);
    if (dolarFile) formData.append('dolar', dolarFile);
    setUploading(true);
    setUploadStatus('Cargando archivos...');

    fetch(`${BACKEND_URL}/api/upload-source`, {
      method: 'POST',
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setUploadStatus(data.error);
        } else {
          setUploadStatus('Fuentes guardadas. Ejecuta la etapa que necesites.');
          setSources((prev) => ({
            ...prev,
            ...data.saved && {
              training_path: data.saved.training ?? prev.training_path,
              dolar_path: data.saved.dolar ?? prev.dolar_path,
            },
          }));
        }
      })
      .catch(() => setUploadStatus('No se pudo cargar. Reintenta.'))
      .finally(() => setUploading(false));
  };

  const handleRun = (action: ActionType) => {
    if (isRunning) return;
    setCurrentAction(action);
    setStatusMessages([`Preparando ejecución de ${ACTION_LABELS[action].toLowerCase()}.`]);
    setIsRunning(true);
    setProgress(10);
    socket.emit(ACTION_EVENT[action]);
  };

  const progressLabel = isRunning
    ? `Ejecutando ${ACTION_LABELS[currentAction || 'full']}`
    : progress === 100
      ? 'Datos listos y actualizados'
      : 'Listo para ejecutar el pipeline';

  return (
    <div className="page">
      <div className="hero">
        <div className="hero-content">
          <div className="brand">
            <img src={logo} className="logo" alt="Logo Horizonte Inmobiliaria" />
            <div>
              <p className="eyebrow">Panel de Datos</p>
              <h1>Horizonte Inmobiliaria</h1>
              <p className="subtitle">
                Control y actualización del pipeline de datos para el equipo comercial.
              </p>
              <div className="meta-card">
                <p className="card-label small">Última actualización</p>
                <p className="card-value small">{lastExecutionDate}</p>
                <p className="card-hint">Archivo limpio listo para el panel.</p>
              </div>
            </div>
          </div>
          <div className="hero-image" aria-hidden="true" />
        </div>
      </div>

      <main className="content">
        <div className="tabs">
          <button className={`tab-button ${activeTab === 'pipeline' ? 'active' : ''}`} onClick={() => setActiveTab('pipeline')}>
            Pipeline
          </button>
          <button className={`tab-button ${activeTab === 'pricing' ? 'active' : ''}`} onClick={() => setActiveTab('pricing')}>
            Pricing
          </button>
          <button className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            Dashboard
          </button>
          <button className={`tab-button ${activeTab === 'chatbot' ? 'active' : ''}`} onClick={() => setActiveTab('chatbot')}>
            Chatbot
          </button>
          <button className={`tab-button ${activeTab === 'docs' ? 'active' : ''}`} onClick={() => setActiveTab('docs')}>
            Documentación
          </button>
        </div>
        {activeTab === 'pipeline' && (
          <>
        <section className="split">
          <div className="card upload">
            <p className="card-label">Fuentes de datos</p>
            <h2>Configurar archivos CSV</h2>
            <p className="card-hint">
              Sube el CSV de propiedades y el de cotización del dólar. Usaremos estos archivos en las próximas ejecuciones.
            </p>
            <div className="file-row">
              <label>CSV de propiedades (entrenamiento)</label>
              <input
                type="file"
                accept=".csv"
                onChange={e => setTrainingFile(e.target.files?.[0] || null)}
              />
              <p className="mini">
                Actual: {sources.training_path ? sources.training_path : 'por defecto en Datasets crudos'}
              </p>
            </div>
            <div className="file-row">
              <label>CSV de dólar</label>
              <input
                type="file"
                accept=".csv"
                onChange={e => setDolarFile(e.target.files?.[0] || null)}
              />
              <p className="mini">
                Actual: {sources.dolar_path ? sources.dolar_path : 'por defecto en Datasets crudos'}
              </p>
            </div>
            <button
              className="secondary-button"
              onClick={handleUploadSources}
              disabled={uploading}
            >
              {uploading ? 'Cargando...' : 'Guardar fuentes'}
            </button>
            <p className="mini status">{uploadStatus}</p>
          </div>

          <div className="card action">
            <div className="action-header">
              <div>
                <p className="card-label">Pipeline de limpieza</p>
                <h2>Ejecutar proceso</h2>
                <p className="card-hint">
                  Limpia datos, convierte precios a USD y elimina valores fuera de mercado.
                </p>
              </div>
              <div className="button-group">
                <button onClick={() => handleRun('clean')} disabled={isRunning} className="secondary-button">
                  {isRunning && currentAction === 'clean' ? 'Procesando...' : 'Solo limpieza'}
                </button>
                <button onClick={() => handleRun('convert')} disabled={isRunning} className="secondary-button">
                  {isRunning && currentAction === 'convert' ? 'Procesando...' : 'Solo conversión'}
                </button>
                <button onClick={() => handleRun('full')} disabled={isRunning} className="run-button">
                  {isRunning && currentAction === 'full' ? 'Procesando...' : 'Pipeline completo'}
                </button>
              </div>
            </div>

            <div className="progress-card">
              <div className="progress-head">
                <div>
                  <p className="card-label">Estado</p>
                  <p className="progress-title">{progressLabel}</p>
                </div>
                <span className="pill">{progress}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="card-hint">
                {isRunning
                  ? 'Puedes seguir trabajando: te avisaremos cuando esté listo.'
                  : 'Elige qué etapa ejecutar según lo que necesites actualizar.'}
              </p>
            </div>
          </div>
        </section>
        <section className="status-section">
          <div className="section-header">
            <div>
              <p className="card-label">Actividad</p>
              <h3>Detalle del proceso</h3>
              <p className="card-hint">
                Mensajes en lenguaje de negocio para seguir el avance paso a paso.
              </p>
            </div>
          </div>

          <div className="status-log" ref={statusLogRef}>
            {statusMessages.length === 0 ? (
              <p className="placeholder-text">Aún no hay ejecuciones. Inicia alguna etapa para ver el detalle.</p>
            ) : (
              statusMessages.map((msg, index) => (
                <div key={index} className="log-message">
                  <span className="dot" />
                  <p>{msg}</p>
                </div>
              ))
            )}
          </div>
        </section>
        </>
        )}

        {activeTab === 'pricing' && (
        <section className="pricing-section">
          <div className="section-header pricing-header">
            <div>
              <p className="card-label">Modelo de precios</p>
              <h3>¿A qué precio debo publicar una propiedad?</h3>
              <p className="card-hint">
                Evita fijar precios solo por percepción: el dataset se arma con variables de ubicación, metraje y mercado para alimentar el modelo de regresión / random forest.
              </p>
            </div>
            <div className="kpi-card">
              <p className="card-label small">Precision modelo</p>
              <p className="card-value small">
                MAE: {typeof trainingModel.metrics?.mae === 'number' ? `USD ${trainingModel.metrics.mae.toFixed(0)}` : '-'}
              </p>
              <p className="card-value small">
                MAPE: {typeof trainingModel.metrics?.mape_pct === 'number' ? `${trainingModel.metrics.mape_pct.toFixed(2)}%` : '-'}
              </p>
              <p className="mini">Actualiza al reentrenar para ver la precision vigente.</p>
            </div>
          </div>

          <div className="pricing-grid">
            <div className="card pricing-story">
              <p className="card-label">Problemática</p>
              <h4>Decisiones de precio subjetivas</h4>
              <p className="card-hint">
                Los agentes suelen fijar precios de publicación por percepción o pedido del propietario, lo que genera ineficiencias.
              </p>
              <ul className="mini-list">
                <li>Propiedades sobrevaluadas que no se venden.</li>
                <li>Publicaciones subvaluadas que reducen margen o comisión.</li>
                <li>Rotación ineficiente del inventario.</li>
              </ul>
            </div>



            <div className="card pricing-actions">
              <p className="card-label">Dataset del modelo</p>
              <h2>Generar dataset de pricing</h2>
              <p className="card-hint">
                Construye el CSV con las variables clave para entrenar el modelo y calcular el precio publicado recomendado.
              </p>
              <div className="feature-chips">
                <span className="chip">Ubicación (barrio + coordenadas)</span>
                <span className="chip">Superficie total y cubierta</span>
                <span className="chip">Ambientes y tipo de propiedad</span>
                <span className="chip">Índice MI_DAN_AX03</span>
                <span className="chip">Precio USD y $/m² publicado</span>
              </div>
              <div className="dataset-meta">
                <div>
                  <p className="mini">Última generación</p>
                  <p className="stat-value">{pricingDataset.last_updated || 'Aún no generado'}</p>
                </div>
                <div>
                  <p className="mini">Filas útiles</p>
                  <p className="stat-value">
                    {pricingDataset.rows !== null && pricingDataset.rows !== undefined
                      ? pricingDataset.rows.toLocaleString()
                      : 'Sin dato'}
                  </p>
                </div>
              </div>
              <p className="mini">Destino: {pricingDataset.path || 'backend/pricing_dataset.csv'}</p>
              <button onClick={() => handleRun('pricing')} disabled={isRunning} className="run-button">
                {isRunning && currentAction === 'pricing' ? 'Procesando...' : 'Crear dataset de pricing'}
              </button>
              <button onClick={handleRetrainModel} disabled={trainingModel.loading} className="secondary-button">
                {trainingModel.loading ? 'Entrenando modelo...' : 'Reentrenar modelo'}
              </button>
              <p className="mini status">{trainingModel.message}</p>
              <p className="mini status">
                Ejecuta antes el pipeline completo para asegurar datos limpios y actualizados.
              </p>
            </div>

            <div className="card pricing-simulator">
              <p className="card-label">Simulador</p>
              <h2>Calcular precio recomendado</h2>
              <p className="card-hint">Ingresa los datos clave de la propiedad. El campo "Precio que planeo publicar" es opcional para ver el desvío.</p>

              <div className="form-grid">
                <label>
                  Tipo de propiedad
                  <select value={predictForm.tipo_propiedad} onChange={e => handlePredictChange('tipo_propiedad', e.target.value)}>
                    {PROPERTY_TYPES.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Operación
                  <select value={predictForm.tipo_operacion} onChange={e => handlePredictChange('tipo_operacion', e.target.value)}>
                    {OPERATIONS.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Ambientes
                  <input type="number" value={predictForm.ambientes} onChange={e => handlePredictChange('ambientes', e.target.value)} />
                </label>
                <label>
                  Dormitorios
                  <input type="number" value={predictForm.dormitorios} onChange={e => handlePredictChange('dormitorios', e.target.value)} />
                </label>
                <label>
                  Baños
                  <input type="number" value={predictForm.banios} onChange={e => handlePredictChange('banios', e.target.value)} />
                </label>
                <label>
                  Superficie total (m²)
                  <input type="number" value={predictForm.superficie_total} onChange={e => handlePredictChange('superficie_total', e.target.value)} />
                </label>
                <label>
                  Superficie cubierta (m²)
                  <input type="number" value={predictForm.superficie_cubierta} onChange={e => handlePredictChange('superficie_cubierta', e.target.value)} />
                </label>
                <label>
                  Provincia
                  <select value={predictForm.provincia} onChange={e => handlePredictChange('provincia', e.target.value)}>
                    <option value="">Selecciona</option>
                    {PROVINCIAS.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Partido
                  <select value={predictForm.partido} onChange={e => handlePredictChange('partido', e.target.value)}>
                    <option value="">Selecciona</option>
                    {PARTIDOS.map(opt => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Localidad (opcional)
                  <input value={predictForm.localidad} onChange={e => handlePredictChange('localidad', e.target.value)} />
                </label>
                <label>
                  Latitud
                  <input type="number" value={predictForm.latitud} onChange={e => handlePredictChange('latitud', e.target.value)} placeholder="-34.60" />
                </label>
                <label>
                  Longitud
                  <input type="number" value={predictForm.longitud} onChange={e => handlePredictChange('longitud', e.target.value)} placeholder="-58.44" />
                </label>
                <label>
                  Fecha de publicación
                  <input type="date" value={predictForm.fecha_publicacion} onChange={e => handlePredictChange('fecha_publicacion', e.target.value)} />
                </label>
                <label>
                  Precio que planeo publicar (USD, opcional)
                  <input type="number" value={predictForm.precio_publicado} onChange={e => handlePredictChange('precio_publicado', e.target.value)} />
                </label>
              </div>

              <button className="run-button" onClick={handlePredict} disabled={predictLoading}>
                {predictLoading ? 'Calculando...' : 'Calcular precio'}
              </button>

              {predictResult && (
                <div className="result-box">
                  <p className="card-label">Resultado</p>
                  <h3 className="stat-value">USD {predictResult.predicted_price.toLocaleString()}</h3>
                  <p className="mini">Rango sugerido: USD {predictResult.price_min.toLocaleString()} - {predictResult.price_max.toLocaleString()}</p>
                  {predictResult.delta_vs_publicado_pct !== null && predictResult.delta_vs_publicado_pct !== undefined && (
                    <p className="mini">
                      Diferencia vs tu precio: {predictResult.delta_vs_publicado_pct > 0 ? '+' : ''}{predictResult.delta_vs_publicado_pct}%
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>
        )}

        {activeTab === 'dashboard' && (
          <section className="dashboard-section">
            <div className="section-header dashboard-header">
              <div>
                <p className="card-label">Power BI</p>
                <h3>Dashboard de propiedades</h3>
                <p className="card-hint">Visualización embebida para seguir inventario, demanda y precios.</p>
              </div>
              <button className="secondary-button" onClick={handleDashboardFullscreen}>
                Pantalla completa
              </button>
            </div>
            <div className="dashboard-embed">
              <iframe
                title="Reporte_Propiedades"
                src="https://app.powerbi.com/reportEmbed?reportId=fe954fb7-74f8-4415-acca-5ae9e833df6d&autoAuth=true&ctid=344979d0-d31d-4c57-8ba0-491aff4acaed"
                frameBorder="0"
                allow="fullscreen"
                allowFullScreen
                ref={dashboardFrameRef}
              />
            </div>
            <p className="mini status">Si no ves el dashboard, valida tus permisos en Power BI o abre en una pestaña separada.</p>
          </section>
        )}

        {activeTab === 'chatbot' && (
          <section className="chatbot-section">
            <div className="section-header chatbot-header">
              <div>
                <p className="card-label">Asistente</p>
                <h3>Chatbot de soporte</h3>
                <p className="card-hint">Interactua con el asistente para consultas del pipeline y datos.</p>
              </div>
              <button className="secondary-button" onClick={handleChatbotFullscreen}>
                Pantalla completa
              </button>
            </div>
            <div className="chatbot-embed">
              <iframe
                title="Chatbot PowerApps"
                src="https://apps.powerapps.com/play/9ca54f1a-b1af-4414-a59e-cd0edc657ab4?source=iframe"
                frameBorder="0"
                allow="fullscreen"
                ref={chatbotFrameRef}
              />
            </div>
            <p className="mini status">Si no carga, abre en una pestana nueva o revisa tus permisos de acceso.</p>
          </section>
        )}

        {activeTab === 'docs' && (
          <section id="docs" className="docs-section">
            <div className="section-header">
              <div>
                <p className="card-label">Documentación</p>
                <h3>Cómo usamos los datos y el modelo</h3>
                <p className="card-hint">Manual extendido para negocio y data, con índice navegable.</p>
              </div>
            </div>
            <div className="doc-index">
              <span>Índice:</span>
              <button onClick={() => scrollToDoc('doc-business')} className="doc-link">Guía para negocio</button>
              <button onClick={() => scrollToDoc('doc-data')} className="doc-link">Modelo y features</button>
              <button onClick={() => scrollToDoc('doc-metrics')} className="doc-link">Métricas e interpretación</button>
              <button onClick={() => scrollToDoc('doc-checklist')} className="doc-link">Checklist antes de publicar</button>
            </div>
            <div className="docs-grid">
              <div id="doc-business" className="card docs-card">
                <p className="card-label">Para negocio</p>
                <h4>Ruta completa del pipeline</h4>
                <p className="doc-text">
                  1) Ingesta: sube el CSV de propiedades y el histórico de dólar (o usa los que ya están). El panel conserva la última versión y la reutiliza si no cargas otra para que todas las corridas usen la misma base.
                </p>
                <p className="doc-text">
                  2) Limpieza: quitamos columnas irrelevantes, corregimos lat/long y filtramos nulos en campos críticos (ubicación, metros, precio, tipo). Así evitamos entrenar con datos incompletos o mal geolocalizados.
                </p>
                <p className="doc-text">
                  3) Conversión: llevamos todos los precios a USD con la cotización de la fecha de publicación y eliminamos outliers extremos en alquiler/venta. 4) Dataset de pricing: derivamos precio USD/m², ratio de superficie cubierta y cruzamos MI_DAN_AX03 por mes y ambientes para capturar el contexto de mercado.
                </p>
                <p className="doc-text">
                  5) Uso operativo: reentrena el modelo y usa el simulador para validar el precio antes de publicar. Si el delta es alto, ajusta o documenta la excepción (amenities premium, vista, ubicación única). Define una banda (ej. ±8%) según estrategia de rotación y margen.
                </p>
              </div>
              <div id="doc-data" className="card docs-card">
                <p className="card-label">Para data</p>
                <h4>Modelo, features y entrenamiento</h4>
                <p className="doc-text">
                  Modelo: GradientBoostingRegressor (scikit-learn). Preprocesamiento: imputación (mediana en numéricos, valor frecuente en categóricos) y OneHotEncoder en categorías. Features: latitud, longitud, ambientes, dormitorios, baños, superficie_total, superficie_cubierta, ratio_cubierta, tipo_propiedad, tipo_operacion, provincia, partido, mi_dan_ax03. Target: precio_dolares.
                </p>
                <p className="doc-text">
                  Entrenamiento: split 80/20 aleatorio. El modelo se guarda en backend/pricing_model.pkl y se recarga automáticamente al predecir. Si agregas columnas en el dataset de pricing, reentrena con el botón para refrescar el pipeline de features.
                </p>
                <p className="doc-text">
                  Hiperparámetros actuales: GradientBoostingRegressor con valores por defecto y random_state=42 para reproducibilidad (backend/server.py, _train_pricing_model): n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, max_features=None, min_samples_split=2, min_samples_leaf=1, loss="squared_error", etc. Preprocesado: imputación mediana en numéricos y más frecuente en categóricos + OneHotEncoder(handle_unknown="ignore"), seguido de un split 80/20 (train_test_split(test_size=0.2, random_state=42)). Motivo: baseline estable y poco propenso al sobreajuste sin tuning, con profundidad baja y 100 árboles estándar; reproducibilidad con random_state; no se ajustaron hiperparámetros adicionales para mantener simplicidad y tiempos de entrenamiento cortos.
                </p>
                <div className="value-box">
                  <p className="card-label">Reglas de limpieza previas</p>
                  <p className="card-hint small">
                    - Se filtran precios objetivo: sólo precio_dolares &gt; 0. Ventas entre 20k–2M USD; alquileres entre 100–10k USD.<br />
                    - Se deduplican columnas del CSV y se seleccionan features sin repetir nombres.<br />
                    - El MAPE se calcula sólo con y_true ≥ 10,000 USD para evitar explosiones por valores muy bajos.
                  </p>
                </div>
              </div>
              <div id="doc-metrics" className="card docs-card">
                <p className="card-label">Interpretación</p>
                <p className="doc-text">
                  M�tricas reportadas: MAE (error absoluto medio en USD) y MAPE (desv�o relativo %). El KPI de negocio (? precio publicado vs modelo) muestra qu� tan alineado est� el precio de salida con la recomendaci�n del modelo.
                </p>
              </div>
              <div id="doc-checklist" className="card docs-card">
                <p className="card-label">Checklist</p>
                <h4>Antes de publicar</h4>
                <p className="doc-text">
                  1) Ejecuta pipeline completo con fuentes actualizadas. 2) Genera dataset de pricing y reentrena el modelo. 3) Simula el precio objetivo y revisa el delta. 4) Define banda de negociación (±8% sugerido) y mensaje comercial. 5) Si el delta es alto, ajusta o documenta la excepción (ej. vista al río, amenities premium).
                </p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
