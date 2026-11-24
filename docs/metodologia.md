# Metodología CRISP-DM y modelos

## Comprensión del negocio

El objetivo experimental es clasificar la calidad visual de un producto individual en `Good`, `Regular` o `Bad`, y estimar su tamaño relativo a partir de una imagen. Es una ayuda de inspección, no una certificación de inocuidad ni un reemplazo de la revisión humana. La salida de tamaño actualmente se expresa en píxeles normalizados; no es una medida física.

## Comprensión y preparación de los datos

La fuente pública Fruit Quality Classification se descarga fuera de Git en `data/raw/kaggle`. La colección propia se organiza en `data/raw/own_collection/<Good|Regular|Bad>/<fruit>/`. Las escenas Mixed se segmentan para proponer recortes; los candidatos permanecen sin supervisión y no entran al modelo hasta que su fila en `data/annotations/mixed_candidates_annotations.csv` tenga una calidad revisada. Las reglas de anotación, madurez, defectos, privacidad y medidas de referencia están en [protocolo_anotacion.md](protocolo_anotacion.md).

Cada registro conserva `sample_id`, `source`, `source_image_id` y `group_id`. Los recortes de una misma escena comparten grupo y deben quedar en una sola partición. La preparación crea aproximadamente 70/15/15 para train/validation/test con semilla 42, falla si una clase presente desaparece de alguna partición o si hay fuga de grupos, y exige al menos tres ejemplos de cada clase presente en train para la validación cruzada de tres folds. La proporción exacta puede variar al asignar grupos completos; se reportan los conteos realmente obtenidos.

Las imágenes se redimensionan a 128 × 128, se aplica CLAHE al canal L en LAB y se guardan en los directorios de cada split. Para los modelos clásicos, las imágenes procesadas producen 108 variables:

- 96 componentes de histogramas HSV: 32 bins por canal, normalizados por histograma.
- 6 estadísticas RGB: media y desviación estándar de cada canal, divididas por 255.
- 6 estadísticas HSV: media y desviación estándar de cada canal, divididas por 255.

Los valores faltantes/no finitos y las dimensiones se verifican antes del entrenamiento. La clase ausente no se rellena artificialmente: con soporte cero, un experimento de tres clases no está definido.

## Modelos y fundamento técnico

### Random Forest

El ensamble reduce la varianza de árboles individuales y captura relaciones no lineales entre estadísticas de color. Usa `class_weight="balanced"` y semilla 42. La búsqueda en rejilla explora `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 20, 40}` y `min_samples_split ∈ {2, 5}`. La selección usa Macro-F1 en validación cruzada estratificada y consciente de grupos.

### SVM

Una SVM busca una frontera de margen máximo; el kernel lineal ofrece una frontera lineal y RBF permite fronteras no lineales. El pipeline ajusta `StandardScaler` dentro de cada fold para evitar fuga de estadísticas. La búsqueda explora `C ∈ {0.1, 1, 10}`, kernel `{linear, rbf}` y `gamma ∈ {scale, auto}`, con pesos balanceados y probabilidades para AUC cuando esté definido. La selección usa Macro-F1 y los mismos grupos/folds que Random Forest.

### CNN

La CNN existente usa tres bloques convolucionales de 32, 64 y 128 filtros, normalización por lotes, max-pooling y dropout 0.25; después una capa densa de 256 unidades, Batch Normalization, dropout 0.5 y softmax. Se optimiza con Adam (tasa inicial `10⁻³`) y entropía cruzada categórica dispersa:

\[
L = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i}\log p(y_i\mid x_i)
\]

Los pesos `w` compensan el desbalance de train. El aumento de datos aplica rotación, volteo horizontal, zoom, cambios pequeños de brillo y traslación. Se entrenan hasta 15 épocas; `EarlyStopping` vigila `val_loss` (paciencia 7), `ReduceLROnPlateau` reduce la tasa tras 3 épocas sin mejora y se conserva el checkpoint con mejor `val_accuracy`. El conjunto de validación guía estas decisiones; test no participa. La configuración actual no constituye una búsqueda exhaustiva de hiperparámetros de CNN y debe describirse como tal.

### Línea base

`DummyClassifier(strategy="most_frequent")` aprende únicamente la clase más común de train. Su Macro-F1 y exactitud de validación/test permiten comprobar si un modelo aprende más que la distribución mayoritaria.

## Métricas y reglas de evaluación

Sea `C` el número de clases y `TP_c`, `FP_c`, `FN_c` los conteos de la clase `c`:

- Exactitud: \(\sum_c TP_c/N\).
- Exactitud balanceada: promedio del recall por clase.
- Macro-Precision: \(\frac{1}{C}\sum_c TP_c/(TP_c+FP_c)\).
- Macro-Recall: \(\frac{1}{C}\sum_c TP_c/(TP_c+FN_c)\).
- Macro-F1: promedio no ponderado de los F1 por clase; trata cada clase con igual importancia y hace visible el bajo desempeño en clases minoritarias.
- AUC-ROC: se calcula desde probabilidades solo cuando el test contiene todas las clases configuradas y las columnas están alineadas con el mapa de etiquetas. En otro caso se informa NaN con su motivo, nunca cero.

Los notebooks 03/04 guardan parámetros, tiempos, soporte, informes por clase, matrices de confusión absolutas/normalizadas y figuras vectoriales SVG más PNG. El modelo se escoge usando validación; el test se evalúa después y no se reutiliza para ajuste. El análisis desglosa errores por calidad, fruta y fuente y conserva ejemplos de falsos Good/falsos Bad. Si Regular tiene soporte cero, los resultados Good/Bad son piloto binario, no una evaluación final de tres clases ni una comparación equivalente de clases.

## Estimación de tamaño y límites

El estimador actual crea una máscara HSV sencilla, toma el contorno externo de mayor área y calcula:

\[
d_{\mathrm{norm}} = \frac{\max(\text{ancho},\text{alto del contorno})}
{\max(\text{ancho},\text{alto de la imagen})}.
\]

Asigna `small` si \(d_{\mathrm{norm}}<0.35\), `medium` si está entre 0.35 y 0.65, y `large` en caso contrario. Este cociente depende de distancia, perspectiva, encuadre, segmentación y fondo; no permite inferir milímetros sin un patrón de escala y calibración. Para validar tamaño físico, anotar `diameter_mm` con regla/calibre y añadir una salida calibrada en mm antes de reportar MAE/RMSE. Las etiquetas manuales de tamaño se preservan frente a la heurística, pero no constituyen por sí mismas una predicción evaluada.

## Estado de evidencia

Las conclusiones numéricas se incorporan después de ejecutar los cuadernos con imágenes, anotaciones y medidas verificadas. La generalización, el desempeño final y cualquier reducción de desperdicio se reportan únicamente cuando existe evidencia medida.