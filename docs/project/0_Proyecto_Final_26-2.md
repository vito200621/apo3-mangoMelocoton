# Proyecto final

## Algoritmos y Programación III

### Semestre 2026-2

**Facultad de Ingeniería, Diseño y Ciencias Aplicadas**  
**Departamento de Computación y Sistemas Inteligentes**  
**Ingeniería de Sistemas**

---

## Lineamientos para el proyecto final

El proyecto final del curso Algoritmos y Programación III es una actividad grupal (3 estudiantes por grupo) que busca desarrollar una solución a un problema real utilizando modelos de analítica y conjuntos de datos de diversos formatos. Cada grupo deberá comprender el problema, investigar su contexto y antecedentes, definir una metodología de trabajo y proponer métricas de desempeño adecuadas para evaluar el progreso.

Se espera que, a lo largo del desarrollo, se entrenen y evalúen diferentes modelos de analítica. Para cada modelo, se deben ajustar adecuadamente los hiperparámetros y evaluar los resultados con base en métricas predefinidas. Cada grupo deberá utilizar la metodología CRISP-DM, adaptándola a las necesidades particulares de su proyecto.

## 1. Caso de estudio propuesto: Clasificador de calidad y tamaño de frutas y verduras

### 1.1. Contexto

En mercados y agroindustrias, la clasificación manual de productos frescos (tomates, manzanas, papas, etc.) según su tamaño, madurez o defectos visibles es un proceso lento, subjetivo y propenso a errores. Esta situación genera pérdidas económicas, desperdicio de alimentos y falta de estandarización.

Se requiere desarrollar un sistema automático de clasificación de calidad basado en visión por computadora, capaz de analizar imágenes de frutas o verduras, asignar una categoría de calidad y estimar el tamaño relativo del producto. El sistema deberá ser entrenado con un conjunto de datos que combine fuentes públicas y recolección propia, y será evaluado mediante métricas rigurosas.

### 1.2. Entrada

Imágenes estáticas (fotografías tomadas con cámara web o celular) de una fruta o verdura individual, sobre un fondo simple y uniforme.

### 1.3. Salidas

- **Clase de calidad:** 3 o más categorías definidas por el grupo.
- **Estimación de tamaño:** pequeño, mediano o grande, o diámetro en píxeles normalizados.

### 1.4. Formato de despliegue

Interfaz gráfica simple (por ejemplo, con Tkinter, PyQt o una aplicación web con Streamlit) que permita cargar una imagen o capturarla en tiempo real usando la cámara y que muestre la predicción obtenida.

### 1.5. Opcional (extensión)

Simulación de una línea de empaque en tiempo real con cámara y Raspberry Pi.

## 2. Datos

### 2.1. Base de datos de referencia

Se utilizará el conjunto de datos *Fruit Quality Classification*, disponible en:

<https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification>

Para la evaluación o el enriquecimiento de la base de datos, se puede emplear la carpeta **mixed quality**, que contiene imágenes de diferentes frutas con distintas calidades. Los estudiantes deberán segmentar individualmente cada fruta presente en las imágenes (si hay varias por foto) para obtener ejemplares individuales.

### 2.2. Coordinación grupal y entre toda la clase

Cada grupo deberá recolectar entre 30 y 50 imágenes adicionales de frutas o verduras reales, obtenidas en plazas de mercado, supermercados o en sus propios hogares. Las imágenes deben abarcar diferentes estados de madurez, tamaños y defectos (golpes, manchas y podredumbre).

### 2.3. Anotación manual

Cada grupo debe etiquetar sus propias imágenes con la categoría de calidad y el tamaño correspondiente. Las etiquetas deben ser coherentes con las definiciones establecidas para el proyecto.

Entre todos los estudiantes se definirá un mecanismo para compartir anotaciones entre los grupos (por ejemplo, una carpeta compartida en la nube o un repositorio común), de modo que al final se disponga de un conjunto de datos más grande y diverso.

## 3. Materiales y métodos a utilizar

Los estudiantes deberán elegir **al menos dos modelos distintos de *machine learning* y uno de *deep learning***, entre los siguientes:

- **Modelos tradicionales de *machine learning*:** regresión logística, K-vecinos más cercanos (KNN), máquinas de soporte vectorial (SVM), *Bagging*, árboles de decisión, *Random Forest* y XGBoost. Se deben ajustar los hiperparámetros mediante validación cruzada o búsqueda en rejilla.
- **Redes neuronales convolucionales (CNN) simples:** arquitecturas pequeñas (por ejemplo, 2 o 3 capas convolucionales + *pooling* + capas densas). Se permite el uso de *transfer learning*, siempre que se congelen capas y se añadan capas propias, aunque se recomienda priorizar modelos entrenados desde cero para un mejor aprendizaje.
- **Redes neuronales totalmente conectadas (MLP):** si se trabaja con características extraídas manualmente.

## 4. Metodología de trabajo: CRISP-DM

Cada grupo debe documentar explícitamente cada fase de CRISP-DM adaptada a su proyecto. No basta con copiar el diagrama; se debe mostrar evidencia de trabajo en cada fase:

1. **Comprensión del negocio:** ¿Por qué es importante clasificar la calidad? ¿Qué impacto económico o social tiene?
2. **Comprensión de los datos:** análisis exploratorio (distribución de clases, desbalanceo, calidad de las imágenes y variabilidad).
3. **Preparación de los datos:** procesamiento y manejo de clases desbalanceadas (si aplica).
4. **Modelado:** entrenamiento, ajuste de hiperparámetros y selección del modelo.
5. **Evaluación:** métricas en el conjunto de prueba, análisis de errores y comparación con una línea base.
6. **Despliegue:** desarrollo de una interfaz gráfica sencilla que permita al usuario ver en tiempo real la calidad de una fruta presentada ante la cámara.

Se debe entregar un **diagrama de flujo personalizado** que refleje la aplicación concreta de CRISP-DM.

## 5. Evaluación

Se evaluará la calidad del trabajo mediante las siguientes preguntas:

- ¿La metodología es clara y robusta?
- ¿Las aproximaciones realizadas en el proyecto son razonables?
- ¿Los datos se exploraron y procesaron de forma adecuada?
- ¿Las soluciones propuestas son ingeniosas e interesantes?
- ¿Se explican correctamente los impactos de la solución en el contexto abordado?
- ¿Se complementaron los datos iniciales?
- ¿Los estudiantes desarrollaron y transmitieron conocimientos no triviales sobre el problema, los algoritmos y los modelos?
- ¿El trabajo demuestra el desarrollo de las competencias definidas para este curso?

Se compartirá un documento para que un representante de cada grupo informe los integrantes y el enlace al repositorio de entrega en **GitHub**. El repositorio debe tener una estructura clara, organizada según las fases definidas en la metodología.

## 6. Aspectos a tener en cuenta

Se debe detallar el problema, la metodología, las métricas para medir el progreso, los datos recolectados, el análisis exploratorio de los datos y los siguientes pasos o etapas del proyecto.

También se debe incluir un análisis de los aspectos éticos relevantes al implementar soluciones de IA en el contexto del problema abordado.

Se debe detallar el entrenamiento de los modelos (incluyendo el ajuste de hiperparámetros), los resultados obtenidos (métricas, gráficas, etc.) y el plan de despliegue. Además, se realizará un análisis inicial de los impactos de la solución en el contexto del problema.

Se espera que el nivel de profundidad del análisis y la calidad de los resultados sean rigurosos. Al final, se deberá presentar el análisis final de los impactos de la solución en el contexto abordado.

Al concluir el proyecto, se debe incluir un video corto de no más de 10 minutos que presente el proyecto, el contexto del problema, las técnicas utilizadas, los resultados y los principales logros alcanzados.

Es fundamental que el código fuente esté bien documentado. Si se utilizan datos o código fuente de terceros, deben referenciarse de forma clara y explícita; de lo contrario, se considerará fraude.

Los informes deben contener explicaciones claras y concisas. Se recomienda incluir diagramas de flujo, diagramas de bloques u otras figuras que ilustren la metodología, la arquitectura de software y los resultados, tanto en el informe como en la presentación. Se deben utilizar gráficos de calidad vectorial.

## 7. Estructura básica del informe final

**Extensión máxima: 7 páginas.**

1. **Título.**
2. **Resumen (Abstract).**
3. **Introducción:** contexto, descripción del problema y justificación de su interés.
4. **Fundamentos teóricos:** ¿Qué debe saber el lector para comprender el desarrollo?  
   **Nota:** seleccionar cuidadosamente el contenido de esta sección, evitando generalidades.
5. **Metodología:** ¿Cómo se abordó el proyecto?  
   **Nota:** no se busca una copia exacta del diagrama CRISP-DM.
6. **Resultados:** ¿Cómo se desempeñaron los modelos en diferentes conjuntos de datos? Incluir métricas de interés para el problema específico.
7. **Análisis de resultados:** ¿Qué se observa en los resultados? ¿Los modelos generalizan bien? ¿Hay sobreajuste (*overfitting*)? ¿Qué funciona bien? ¿Qué falla? ¿Cómo se comparan los resultados con otros reportados en la literatura?
8. **Conclusiones y trabajo futuro:** ¿Qué se hizo? ¿Qué se aprendió? ¿Qué se puede mejorar?
9. **Referencias bibliográficas:** incluir solo artículos, libros o materiales digitales que hayan sido leídos y utilizados. **Usar formato IEEE.**

### Observaciones

1. Revisar con detenimiento artículos publicados en conferencias de interés como NIPS, ICML o ICLR, entre otras, antes de redactar los informes finales.
2. TODOS los grupos deben prestar mucha atención a la rúbrica de evaluación, que estará presente en INTU, para ajustar su informe final adecuadamente.
