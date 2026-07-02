# ⚽ FIFA Scout

Web app de scouting de jugadores de fútbol construida con **Streamlit**, que despliega el
modelo de predicción de potencial entrenado en el TP2 de la materia *Inteligencia Artificial
y Aprendizaje Automático I* (Licenciatura en Ciencias de Datos).

**App desplegada:** _[completar con la URL de Streamlit Community Cloud una vez desplegada]_

## El proyecto

Este es el TP4 (cierre) de un proyecto integrador de 4 etapas sobre el dataset **FIFA
Players** (17.699 jugadores, 73 columnas):

| Etapa | Descripción | Estado |
|---|---|---|
| TP1 | Limpieza y feature engineering → `fifa_players_model_ready.csv` | ✅ |
| TP2 | Regresión para predecir `potential` (techo de crecimiento, 1-99) | ✅ |
| TP3 | Clasificación de posición específica del jugador | 🔄 (se integra vía `modelo_clasificador_tp3.joblib`) |
| TP4 | Esta app de scouting | ✅ |

## Cómo fue entrenado el modelo (TP2)

Se compararon 5 algoritmos con un split 80/20 (validado empíricamente comparando 60/40,
70/30 y 80/20 antes de fijarlo) sobre 23 features (atributos de juego, scores agregados del
TP1, codificaciones y dummies de posición — sin incluir `potential_gap`, que filtraría el
target):

| Modelo | R² Test | MAE | RMSE | Gap train-test |
|---|---|---|---|---|
| Regresión Lineal | 0.8328 | 1.9558 | 2.4794 | 0.05% |
| Árbol de Decisión | 0.9200 | 1.1234 | 1.7151 | 0.98% |
| KNN (7 features, K=9) | 0.9029 | 1.3427 | 1.8898 | 2.14% |
| Random Forest (base) | 0.9301 | 1.0205 | 1.6033 | 5.99% |
| **GB Optimizado (GridSearchCV)** ✅ | **0.9355** | **0.9973** | **1.5402** | **2.23%** |

El modelo ganador es un `sklearn.pipeline.Pipeline` (`StandardScaler` + `GradientBoostingRegressor`)
con hiperparámetros optimizados vía `GridSearchCV`: `learning_rate=0.05`, `max_depth=5`,
`n_estimators=300`, `subsample=0.8`. Fue el que mejor balanceó precisión y generalización
(mejor R²/MAE/RMSE del grupo y gap muy inferior al del Random Forest base).

## Las 5 páginas de la app

1. **🔍 Explorador de Joyas** — scatter interactivo (Plotly) de valor de mercado vs.
   potencial predicho, con filtros por posición, edad, valor, nacionalidad y overall mínimo.
   Incluye una tabla de "joyas ocultas": jóvenes de alto potencial predicho y bajo valor de
   mercado.
2. **⚡ Predictor Individual** — sliders y selectores para cargar manualmente los 23
   atributos de un jugador, predicción de potencial con gauge chart, clasificación textual y
   explicabilidad con **SHAP** (waterfall plot de las variables más influyentes).
3. **📁 Análisis en Lote** — carga masiva de un CSV con múltiples jugadores, validación de
   columnas, predicción en lote y descarga de resultados. Incluye plantilla descargable.
4. **🎯 Clasificador de Posición** — espacio reservado para el modelo del TP3. Si
   `modelo_clasificador_tp3.joblib` no existe, muestra la distribución de posiciones del
   dataset como contexto informativo; si existe, arma la interfaz de clasificación de forma
   automática a partir de `feature_names_in_` del modelo (no requiere tocar el código de la app).
5. **📊 Sobre el Modelo** — comparación de los 5 modelos evaluados, importancia de
   variables calculada en vivo desde el pipeline cargado, distribución de residuos y
   estadísticas del dataset.

Además incluye un contador de predicciones por sesión (`st.session_state`) como monitoreo
básico de uso.

## Cómo correr la app localmente

```bash
# 1. Cloná el repositorio
git clone <url-del-repo>
cd fifa-scout-app

# 2. Creá un entorno virtual e instalá dependencias
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Asegurate de tener modelo_gb_pipeline.joblib y fifa_players_model_ready.csv
#    en esta misma carpeta (ver instrucciones de serialización en el TP2)

# 4. Corré la app
streamlit run app.py
```

La app abre en `http://localhost:8501`.

## Stack tecnológico

- **Python 3.10+**
- **Streamlit** — interfaz web
- **scikit-learn** — Pipeline de predicción (StandardScaler + GradientBoostingRegressor)
- **pandas / numpy** — manejo de datos
- **Plotly** — visualizaciones interactivas
- **SHAP** — explicabilidad del modelo
- **joblib** — serialización del modelo

## Estructura del repositorio

```
fifa-scout-app/
├── app.py                          # App principal (5 páginas)
├── modelo_gb_pipeline.joblib       # Modelo serializado (Pipeline TP2)
├── fifa_players_model_ready.csv    # Dataset procesado (output del TP1)
├── modelo_clasificador_tp3.joblib  # (opcional) Modelo de clasificación del TP3
├── requirements.txt
├── README.md
└── .gitignore
```

## Integrantes del equipo

- _[Nombre completo — completar]_
- _[Nombre completo — completar]_
- _[Nombre completo — completar]_

Licenciatura en Ciencias de Datos — Inteligencia Artificial y Aprendizaje Automático I — 2026
