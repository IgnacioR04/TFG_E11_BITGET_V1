# BOT-E11-XGBOOST

Bot de trading automatizado BTC/USDT Futuros Perpetuos en Bitget.
Estrategia E11: pipeline HMM (regimen) → GARCH proxy (volatilidad) → EMA200 (tendencia) → XGBoost (señal) → Kelly fraccional (sizing).
Solo opera LONG. Se ejecuta cada minuto via GitHub Actions.

## Estructura

```
BOT-E11-XGBOOST/
├── .github/workflows/
│   ├── bot.yml          # Cron cada minuto, ejecuta bot.py
│   └── static.yml       # Despliega docs/ en GitHub Pages
├── docs/
│   ├── index.html       # Dashboard (GitHub Pages)
│   ├── data.json        # Snapshot del estado del bot
│   └── .nojekyll
├── models/
│   ├── xgb_bull.pkl     # Modelo XGBoost BULL (subir manualmente)
│   └── xgb_bear.pkl     # Modelo XGBoost BEAR (no usado por ahora)
├── bot.py               # Logica principal
├── bitget_api.py        # Wrapper API Bitget HMAC-SHA256
├── state.json           # Estado persistente del bot
├── requirements.txt
└── README.md
```

## Setup (6 pasos)

### 1. Crear repo en GitHub
- Repo **privado** (contiene los .pkl)
- Subir todos estos archivos

### 2. Subir modelos
- Crear carpeta `models/`
- Subir `xgb_bull.pkl` y `xgb_bear.pkl`

### 3. Activar GitHub Pages
- Settings → Pages → Source: **GitHub Actions**

### 4. Dar permisos de escritura al workflow
- Settings → Actions → General → Workflow permissions
- Seleccionar **"Read and write permissions"**

### 5. Añadir GitHub Secrets (API keys de Bitget)
- Settings → Secrets and variables → Actions → New repository secret
- `BITGET_API_KEY`
- `BITGET_API_SECRET`
- `BITGET_PASSPHRASE`

### 6. Test manual
- Actions → "E11 XGBoost Bot" → "Run workflow"
- Comprobar que se actualiza `docs/data.json`
- Abrir el dashboard en GitHub Pages

## Parametros de la estrategia

| Parametro       | Valor | Descripcion                     |
|-----------------|-------|---------------------------------|
| DELTA           | 0.20  | Dead zone XGBoost               |
| LEVERAGE        | 5x    | Apalancamiento fijo             |
| BASE_PCT        | 20%   | Sizing minimo Kelly             |
| PCT_MAX         | 55%   | Sizing maximo Kelly             |
| KELLY_DIV       | 3.0   | Divisor Kelly fraccional        |
| GARCH_UMBRAL    | 0.80  | Percentil corte volatilidad     |
| EMA_SPAN        | 200   | Periodo EMA tendencia diaria    |
| ALLOW_SHORTS    | False | Shorts desactivados             |
| COMISION        | 0.04% | Por lado                        |

## Condiciones de parada automatica

El bot se auto-pausa si:
1. Capital < 50% del inicial (drawdown > 50%)
2. 5 trades consecutivos negativos
3. 3 errores API consecutivos
4. Modelo .pkl no cargable

Para reanudar: cambiar `"paused": false` en `state.json` y hacer push.
