# Investigación Académica para la Fase de Comprensión del Negocio (Business Understanding) según Metodología CRISP-DM: Sistema Inteligente de Clasificación de Calidad y Estimación de Tamaño de Productos Hortofrutícolas

## 1. Resumen Ejecutivo

La fase inicial del ciclo de vida de un proyecto de minería de datos y aprendizaje automático, bajo la metodología CRISP-DM (*Cross-Industry Standard Process for Data Mining*), exige establecer una comprensión rigurosa del problema de negocio, determinar sus metas operativas y traducirlas en objetivos técnicos cuantificables. En el sector agroindustrial y de comercialización de alimentos frescos, las ineficiencias asociadas al manejo poscosecha representan una vulnerabilidad crítica para la seguridad alimentaria y la sostenibilidad económica$^{1}$. Tradicionalmente, los procesos de inspección visual, clasificación por grados de calidad y calibrado por tamaño de frutas y verduras se han llevado a cabo mediante evaluación humana manual$^{2}$. Esta modalidad presenta limitaciones operativas estructurales, tales como variabilidad inter-operador, lentitud funcional, fatiga cognitiva y una marcada subjetividad en la emisión de juicios de calidad$^{2}$.

El presente reporte académico desarrolla la fundamentación teórica, sectorial y técnica requerida para la fase de Comprensión del Negocio de un sistema inteligente de inspección visual alimentaria. La solución propuesta integra algoritmos de visión por computadora y aprendizaje automático (*Machine Learning* y *Deep Learning*) orientados a clasificar la calidad de frutas e inferir dimensiones físicas relativas a partir de fotografías digitales individuales$^{2}$. Para ello, se evalúan las bases macroeconómicas del desperdicio alimentario a nivel global, nacional (Colombia) y regional (Valle del Cauca), el marco normativo comercial aplicable (Codex Alimentarius, UNECE e ICONTEC), el estado del arte de la visión artificial aplicada a la agroindustria y la caracterización técnica del conjunto de datos de referencia *FruitNet* (disponible en Kaggle como *Fruit Quality Classification*)$^{1}$. El análisis concluye con la delimitación formal de las oportunidades de negocio, la matización de las capacidades reales de la tecnología basada exclusivamente en espectro visible (RGB) y una hoja de ruta para la mitigación de riesgos éticos y técnicos$^{4}$.

## 2. Contexto Económico y Social del Sector Hortofrutícola

### 2.1 Panorama Internacional de las Pérdidas y Desperdicios de Alimentos

La Organización de las Naciones Unidas para la Alimentación y la Agricultura (FAO) estima que aproximadamente un tercio de la producción mundial de alimentos destinados al consumo humano se pierde o desperdicia a lo largo de la cadena de suministro$^{1}$. En el segmento específico de frutas, hortalizas, raíces y tubérculos, la tasa de pérdida y desperdicio alcanza el 45% de la oferta disponible global$^{1}$. Esta cifra representa el porcentaje de desperdicio más elevado entre todos los grupos alimentarios, superando significativamente a los productos cárnicos y lácteos (20%) y a los pescados (35%)$^{1}$.

En las naciones industrializadas o de altos ingresos, la mayor proporción del desperdicio se concentra en las etapas de distribución minorista (*retail*) y consumo final, motivada por exigencias estéticas comerciales desmedidas y hábitos de compra inadecuados$^{1}$. Por el contrario, en las economías emergentes y en desarrollo, las reducciones de masa alimentaria ocurren predominantemente durante la producción agrícola, el manejo poscosecha, el almacenamiento y el transporte$^{1}$. En estos contextos, la falta de infraestructura de refrigeración, la ausencia de empaques técnicos estandarizados y el predominio de métodos manuales no calibrados de selección aceleran la degradación física y biológica de los productos$^{1}$.

### 2.2 Panorama Nacional en Colombia

En Colombia, la dinámica del sector hortofrutícola es estratégica para el desarrollo rural y la soberanía alimentaria. Sin embargo, la cadena de suministro enfrenta ineficiencias severas. De acuerdo con los diagnósticos institucionales del Departamento Nacional de Planeación (DNP) y la formulación de la política pública CONPES 3866, en el país se pierden y desperdician 9.76 millones de toneladas de alimentos al año, cifra equivalente al 34% de la oferta nacional total$^{1}$.

Al analizar la cadena por grupos alimentarios, las frutas y verduras representan la fracción volumétrica más grande de las pérdidas del país$^{1}$. El DNP y el Ministerio de Salud y Protección Social estimaron una pérdida poscosecha anual de 1,426,932 toneladas de frutas y verduras en las fases tempranas de la cadena (manejo en finca, empaque y transporte inicial)$^{1}$. Esta cifra equivale al 39% de la oferta total de frutas y verduras registrada en dicho periodo$^{1}$. Desglosado por subgrupos, las pérdidas correspondieron a 1,154,923 toneladas de frutas (32% de su oferta) y 272,009 toneladas de verduras (7% de su oferta)$^{1}$.

### 2.3 Panorama Regional: Valle del Cauca y Nodos de Distribución

El departamento del Valle del Cauca constituye un nodo agrologístico e industrial clave en Colombia, caracterizado por una alta productividad en cultivos agroindustriales y una oferta hortofrutícola diversificada en sus distintas franjas térmicas$^{15}$. La región participa activamente en cadenas de exportación y suministro nacional de cítricos (como la mandarina y la naranja), frutas tropicales y hortalizas$^{16}$. Centrales de abasto de escala regional, como la Central de Abastecimientos del Valle del Cauca (CAVASA), junto con las plazas de mercado municipales (e.g., Galería Alameda o Santa Elena en Cali), articulan el flujo de productos desde pequeños y medianos productores del interior del departamento y de zonas aledañas (Cauca, Nariño, Eje Cafetero) hacia el consumo urbano.

En las cadenas de suministro hortofrutícola regionales, las pérdidas poscosecha promedian niveles superiores al 20%$^{15}$. Los pequeños y medianos productores del Valle del Cauca enfrentan asimetrías de información y falta de herramientas tecnológicas en los sitios de acopio. Al no disponer de sistemas automatizados de clasificación por calibre o calidad visual, estos productores comercializan sus cosechas "al bulto" o mediante segregación manual rudimentaria. Esta práctica deriva en la depreciación del producto en las centrales de acopio, donde los intermediarios aplican castigos sobre el precio de compra final argumentando falta de homogeneidad o presencia de defectos heterogéneos en el lote.

| **Dimensión Geográfica** | **Métrica / Indicador Sectorial** | **Valor Registrado / Estimado** | **Fuente Institucional / Referencia** |
| --- | --- | --- | --- |
| **Global** | Desperdicio global de alimentos | 33% de la producción total | FAO$^{1}$ |
| **Global** | Pérdida y desperdicio en frutas y verduras | 45% de la oferta disponible | FAO$^{1}$ |
| **Colombia** | Pérdida y desperdicio total nacional | 9.76 millones ton/año (34%) | DNP / CONPES 3866$^{1}$ |
| **Colombia** | Pérdida poscosecha de frutas y verduras | 1,426,932 ton/año (39% de su oferta) | DNP / MinSalud$^{1}$ |
| **Colombia** | Composición de la pérdida hortofrutícola | 1,154,923 ton (frutas) / 272,009 ton (verduras) | DNP$^{1}$ |
| **Valle del Cauca / Regional** | Pérdidas poscosecha en cadenas regionales | 20% - 30% en promedio poscosecha | UPRA / ADR / EVA$^{15}$ |

## 3. Definición del Problema de Negocio y Deficiencias de la Inspección Manual

### 3.1 Anatomía de la Clasificación Manual

En la mayoría de los centros de empaque (*packhouses*), plazas de mercado y secciones de selección de supermercados en economías en desarrollo, el control de calidad visual de las frutas y verduras es ejecutado por operarios humanos$^{2}$. El procedimiento habitual consiste en la inspección punto por punto de los ejemplares a medida que se desplazan sobre mesas de selección o bandas transportadoras. Los clasificadores separan los productos extrayendo manualmente aquellos que presentan discrepancias geométricas, cambios de coloración indeseados, magulladuras mecánicas, ataques de plagas o signos iniciales de infección fúngica$^{8}$.

### 3.2 Cuellos de Botella Operativos y Subjetividad

La inspección manual presenta deficiencias intrínsecas documentadas en la literatura agroindustrial y de sistemas de producción$^{2}$:

1. **Fatiga Cognitiva y Sensorial**: La repetición de tareas de inspección visual de alta velocidad durante jornadas laborales prolongadas induce a una disminución progresiva de la agudeza visual y de la capacidad atencional. Esto incrementa drásticamente la tasa de error por falso negativo (omitir un fruto defectuoso) y falso positivo (descartar un fruto apto).
1. **Alta Variabilidad Inter- e Intra-Operador**: La percepción de parámetros abstractos como "tamaño mediano", "color característico" o "defecto leve" varía significativamente entre diferentes individuos, e incluso en el mismo operario a diferentes horas del día. La falta de un estándar cuantitativo calibrado provoca que lotes clasificados en distintas turnos de trabajo muestren discrepancias severas en su composición de calidad.
1. **Incapacidad para Cuantificar Defectos Continuos**: El ojo humano es ineficiente para medir objetivamente la proporción porcentual de área afectada por una cicatriz o mancha. Mientras que la normativa exige diferenciar si un defecto cubre menos del 10% o más del 20% de la superficie del fruto$^{20}$, el juzgamiento humano opera mediante estimaciones heurísticas imprecisas.
1. **Vulnerabilidad a Condiciones Ambientales**: Cambios en la temperatura de color de las luminarias del centro de acopio, sombras proyectadas sobre las bandas y contrastes de fondo alteran la percepción humana de la madurez y la pigmentación del fruto$^{3}$.

### 3.3 Impacto Económico Transversal en la Cadena de Valor

Las fallas en el proceso de selección manual repercuten negativamente sobre todos los actores de la cadena hortofrutícola:

- **Agricultores y Productores**: La falta de herramientas de clasificación en origen les impide segregar sus productos en categorías comerciales diferenciadas antes de la venta. Al entregar lotes heterogéneos a mayoristas, reciben un pago equivalente a la categoría inferior contenida en la carga, perdiendo la renta asociada a los frutos de primera calidad.
- **Comerciantes Mayoristas y Distribuidores**: La inclusión inadvertida de frutos con patologías iniciales (e.g., podredumbre blanda o ataques fúngicos invisibles a simple vista) dentro de empaques cerrados acelera la emisión de etileno y la propagación de inóculos hacia frutos sanos adyacentes durante el transporte, destruyendo lotes enteros$^{8}$.
- **Supermercados y Retailers**: Incurren en elevados costos directos por devolución de mercancía rechazada en muelles de recepción debido al incumplimiento de pliegos de condiciones estéticos y dimensiones de calibre exigidas por los clientes finales$^{2}$.
- **Consumidores**: Experimentan insatisfacción por la compra de productos heterogéneos cuyo periodo de vida útil en anaquel es corto e impredecible, estimulando el desperdicio en el hogar$^{1}$.

## 4. Importancia Técnica y Comercial de la Clasificación de Calidad y Tamaño

### 4.1 Caracterización Multidimensional de la Calidad en Alimentos

La calidad de una fruta o verdura no es una propiedad univariada, sino un constructo holístico compuesto por múltiples dimensiones biológicas, químicas y físicas. Para justificar el alcance de un sistema de visión artificial, es imperativo establecer la taxonomía de estas dimensiones y determinar cuáles son factibles de ser estimadas mediante fotometría convencional RGB:

- **Calidad Visual**: Se refiere a los atributos morfológicos y cromáticos externos perceptibles a través de la radiación reflejada en el espectro visible (400–700 nm). Incluye la uniformidad del color de la epidermis, la geometría del fruto, el diámetro ecuatorial o la longitud axial, y la presencia de discontinuidades en la piel (cicatrices, fisuras, manchas, punzamiento de insectos, magulladuras)$^{4}$.
- **Calidad Comercial**: Corresponde al grado de cumplimiento formal de los pliegos de condiciones contractuales o normativos definidos para la transacción mercantil (e.g., Categoría Extra, I, II o Grado Muestra según normas NTC o UNECE)$^{7}$. Incorpora la homogeneidad del lote y los rangos de calibre permitidos.
- **Calidad Organoléptica o Sensorial**: Comprende las propiedades percibidas durante el consumo, como el sabor (balance entre azúcares y ácidos orgánicos), la textura de la pulpa (firmeza, jugosidad, fibrosidad) y el perfil aromático de volátiles$^{10}$.
- **Calidad Nutricional**: Contenido cuantificable de micronutrientes (vitaminas como el ácido ascórbico), minerales, fibra dietética y compuestos bioactivos (polifenoles, carotenoides).
- **Estado de Madurez**: Proceso fisiológico complejo dividido en madurez fisiológica (momento en que el fruto alcanza su pleno desarrollo y puede madurar separado de la planta) y madurez de consumo u organoléptica$^{19}$. En frutos climatéricos (e.g., banano, mango, manzana), la madurez evoluciona activamente poscosecha acompañada de picos respiratorios y síntesis endógena de etileno; en frutos no climatéricos (e.g., fresa, cítricos), la madurez no progresa sustancialmente tras la recolección$^{20}$.
- **Defectos Físicos y Deterioro Biológico**: Lesiones mecánicas (corte, abrasión, magulladura), fisiopatías poscosecha (daño por frío, escaldadura del sol) y descomposición microbiológica (micelios visiblemente desarrollados o reblandecimiento de tejidos por bacterias pectinolíticas)$^{8}$.

| **Dimensión de Calidad** | **Descripción Técnica** | **¿Evaluable mediante Fotografías RGB Convencionales?** | **Exigencia Tecnológica / Instrumentación Complementaria** |
| --- | --- | --- | --- |
| **Calidad Visual** | Geometría, color epidérmico, presencia de manchas y cicatrices externas. | **Sí (Alta factibilidad)** | Algoritmos de segmentación de imagen, espacios de color (HSV/Lab), redes convolucionales$^{2}$. |
| **Estado de Madurez** | Evolución cromática de la piel asociada a la degradación de clorofila y síntesis de pigmentos. | **Parcial (Moderada factibilidad)** | Válido solo cuando la madurez tiene correlación cromática externa directa. Ineficiente si el cambio es interno$^{4}$. |
| **Defectos Físicos / Deterioro** | Lesiones epidérmicas, podredumbre expuesta, abrasiones, daño por insectos visible. | **Sí (Alta factibilidad)** | Clasificación de patrones de textura (GLCM) y detección de objetos mediante Deep Learning$^{2}$. |
| **Tamaño y Calibre** | Diámetro máximo, área proyectada, dimensión axial normalizada. | **Parcial (Condicionado)** | Requiere escala de referencia conocida (marcador fiduciario), distancia focal fija o sensores de profundidad$^{25}$. |
| **Calidad Organoléptica** | Contenido de sólidos solubles (°Brix), acidez titulable, firmeza interna de la pulpa. | **No (Infactible con RGB)** | Requiere espectroscopía de infrarrojo cercano (NIRS), imágenes hiperespectrales o penetrometría$^{10}$. |
| **Calidad Nutricional** | Concentración de vitaminas, capacidad antioxidante, minerales. | **No (Infactible con RGB)** | Requiere métodos analíticos destructivos de laboratorio (HPLC, espectrofotometría). |

### 4.2 Marco Normativo y Estándares Comerciales

El comercio formal de productos frescos se rige por marcos normativos internacionales y nacionales que definen la terminología y las tolerancias estrictas de calidad:

1. **Codex Alimentarius (FAO/OMS)**: Las normas del Codex para frutas y hortalizas frescas (e.g., CODEX STAN 238-2003) establecen disposiciones mínimas de calidad: los productos deben estar enteros, sanos (excluyendo aquellos afectados por podredumbre), limpios, exentos de plagas, exentos de humedad externa anormal y exentos de olores o sabores extraños$^{8}$. Clasifica la calidad en tres categorías estándar: *Categoría Extra* (calidad superior, sin defectos excepto leves modificaciones superficiales), *Categoría I* (defectos leves de forma, coloración o piel que no superen límites estrictos) y *Categoría II* (productos que satisfacen requisitos mínimos pero admiten defectos moderados de forma y piel sin afectación de la pulpa)$^{8}$.
1. **Estándares UNECE (Comisión Económica para Europa de las Naciones Unidas)**: Utilizados ampliamente como referencia en el comercio internacional y por la Unión Europea (e.g., Estándar FFV-50 para manzanas, FFV-35 para fresas, FFV-14 para cítricos)$^{26}$. La UNECE formaliza los niveles de tolerancia por peso o número dentro del empaque; por ejemplo, se admite un límite máximo de tolerancia de calidad del 2% para Categoría Extra, 5% para Categoría I y 10% para Categoría II$^{9}$.
1. **Normativa Colombiana (ICONTEC)**:
    - **NTC 1291**: Establece las generalidades, terminología y requisitos de clasificación para frutas y hortalizas frescas en Colombia, definiendo las categorías *Extra*, *Primera* y *Segunda*, dejando la categoría *Grado Muestra* para productos que no alcanzan los requisitos mínimos pero cuya venta se pacta entre partes$^{7}$.
    - **NTC 4103**: Especificación para la fresa variedad Chandler$^{20}$. Ilustra con precisión los criterios de coloración (escalas cromáticas del 0 al 6 según avance del color rojo) e índices de madurez$^{20}$. Define categorías de calidad limitando los defectos de deformación apical o cicatrices por ácaros al 10% del área total en Categoría I y al 20% en Categoría II$^{20}$.
    - **NTC 5166 y NTC 5422**: Regulan los empaques, dosificación y embalaje comercial para evitar magulladuras y daños mecánicos durante la estiba y transporte de perecederos$^{12}$.

### 4.3 Importancia del Tamaño y Calibre en la Cadena Logística

El calibre de una fruta se determina tradicionalmente por su diámetro ecuatorial máximo (expresado en milímetros) o por su masa individual (expresada en gramos)$^{20}$. El calibre es un determinante directo del valor de mercado: los calibres grandes y medianos suelen acceder a precios *premium* en el mercado de fresco, mientras que los calibres pequeños se canalizan hacia procesos de transformación agroindustrial (jugos, mermeladas, conservas) a precios sensiblemente menores$^{25}$. Asimismo, la uniformidad de calibre dentro de las canastillas o empaques dosificados (NTC 5422) es indispensable para optimizar la densidad de empaque, evitar el aplastamiento de frutos por huelgos excesivos dentro de las cajas y estandarizar las líneas automáticas de sellado y etiquetado$^{12}$.

### 4.4 Diferencia Crítica: Clases del Modelo vs. Categorías Comerciales Oficiales

Un error conceptual recurrente en proyectos de ingeniería de sistemas consiste en asumir que las clases predichas por un modelo de aprendizaje automático (e.g., las clases "Good", "Bad", "Mixed" del dataset *FruitNet*) representan directamente las categorías comerciales legales (*Extra*, *Primera*, *Segunda* de la NTC 1291 o Codex)$^{3}$.

La diferenciación radica en los siguientes aspectos:

- **Modelos de Aprendizaje Automático**: Operan como clasificadores estadísticos de patrones visuales dicotómicos o tricotómicos abstractos. Una etiqueta "Good" en el dataset indica únicamente que el espécimen no presenta necrosis, podredumbre abierta ni deformaciones severas visibles desde el ángulo de captura$^{3}$.
- **Categorías Comerciales Oficiales**: Constituyen certificaciones normativas compuestas que exigen la verificación simultánea de tolerancias porcentuales de área epidérmica afectada, ausencia estricta de residuos químicos, límites microbiológicos, rangos de calibre milimétricos y parámetros internos no visuales$^{8}$.

Por consiguiente, el sistema desarrollado en este proyecto debe entenderse como un **módulo inteligente de pre-clasificación u ordenamiento visual primario**, capaz de segregar productos aptos de deteriorados a alta velocidad, y no como un instrumento sustitutivo de la certificación de inocuidad y grado comercial integral.

## 5. Estado del Arte en Visión por Computadora y Machine Learning Agroindustrial

### 5.1 Pipeline Tradicional de Procesamiento Digital de Imágenes

Históricamente, los sistemas de inspección fotométrica de frutas han empleado técnicas convencionales de procesamiento digital de imágenes compuestas por tres etapas secuenciales$^{4}$:

1. **Preprocesamiento y Segmentación**: Las imágenes capturadas en espacio de color RGB se convierten a espacios alternativos más robustos a las variaciones de iluminación, tales como HSV (Hue, Saturation, Value) o CIE L*a*b*$^{4}$. Mediante umbralización Otsu (e.g., índices combinados como $2R - G - B$) se separa el fondo de la región de interés (RoI) del fruto$^{4}$.
1. **Extracción de Características (Feature Engineering)**:
    - **Color**: Histogramas de color, momentos cromáticos (media, desviación estándar, asimetría en canales H y a*)$^{4}$.
    - **Textura**: Matriz de Coocurrencia de Niveles de Gris (GLCM), de la cual se extraen descriptores estadísticos como contraste, homogeneidad, energía y entropía para detectar rugosidades por bruñido o necrosis epidérmica$^{4}$.
    - **Forma y Geometría**: Descriptores de contorno, excentricidad, compacidad, circularidad y perímetro en píxeles$^{4}$.
1. **Clasificación mediante Aprendizaje Automático Tradicional**: Los vectores de características extraídos manualmente alimentan algoritmos supervisados como *Support Vector Machines* (SVM), *Random Forest*, *K-Nearest Neighbors* (KNN) o *XGBoost*$^{2}$.

### 5.2 Paradigma de Aprendizaje Profundo (Deep Learning) y Redes Convolucionales

El advenimiento de las Redes Neuronales Convolucionales (CNN) transformó el estado del arte al eliminar la necesidad de extracción manual de características. Las capas convolucionales aprenden jerárquicamente representaciones espaciales y cromáticas directas a partir de las imágenes crudas, desde bordes simples y texturas en las capas iniciales hasta patrones morfológicos complejos y lesiones específicas en las capas profundas$^{5}$.

El uso de **Transfer Learning** (aprendizaje por transferencia) sobre arquitecturas preentrenadas en grandes conjuntos de datos (como ImageNet) permite adaptar redes avanzadas a dominios agroindustriales con requerimientos computacionales reducidos. Modelos como ResNet-50, MobileNetV2/V3, VGG-16 e InceptionV3 son frecuentemente seleccionados para estos propósitos$^{5}$.

### 5.3 Evidencia Científica y Estudios Comparativos

Múltiples investigaciones de la literatura científica reciente sustentan la viabilidad y precisión de estos modelos en tareas de evaluación hortofrutícola:

- **Meshram et al. (2021)** desarrollaron el marco *MNet*, diseñado específicamente para mitigar la clasificación errónea de imágenes de frutas al evaluar la calidad de especímenes en entornos con variaciones de fondo e iluminación$^{3}$.
- **Ayaz et al. (2021)** aplicaron extracción de características profundas no lineales combinadas con clasificadores en manzanas, logrando precisiones superiores al 90% en el diagnóstico de enfermedades fisiológicas y patógenas epidérmicas$^{3}$.
- **Momeny et al. (2020)** introdujeron una red CNN basada en un enfoque de *hybrid pooling* para la clasificación precisa de cerezas por calidad visual y tamaño, superando a las arquitecturas tradicionales de agrupamiento (*max/average pooling*) en presencia de solapamiento de clase$^{3}$.
- **Behera et al. (2021)** evaluaron el estado de madurez de frutos de papaya integrando técnicas de aprendizaje automático y transfer learning sobre imágenes RGB, correlacionando de forma precisa la degradación cromática del pericarpio con la fase de maduración$^{3}$.
- **Estudio Benchmark sobre Dataset de Calidad de Frutas (2025)**: Evaluaciones recientes que aplicaron redes CNN convolucionales profundas sobre el conjunto de datos público de frutas de Kaggle (derivado de FruitNet) reportaron una exactitud global del 92.5%$^{2}$. Las precisiones por clase alcanzaron un 93.0% para frutas de alta calidad, 91.7% para calidad media/mixta y 92.3% para frutas de baja calidad o deterioradas$^{2}$. El estudio identificó que la principal tasa de confusión (8%) ocurrió al clasificar especímenes de baja calidad como de calidad media, debido a imperfecciones de iluminación y ángulos donde los defectos epidérmicos quedaban parcialmente ocluidos$^{2}$.

| **Algoritmo / Modelo** | **Requisito de Procesamiento de Datos** | **Ventajas Técnicas** | **Limitaciones Operativas** | **Precisión Reportada en Literatura** |
| --- | --- | --- | --- | --- |
| **SVM (Support Vector Machines)** | Exige extracción previa de vectores de características (color, GLCM, forma)$^{4}$. | Eficiente en espacios de alta dimensión con pocas muestras; bajo costo computacional en inferencia$^{32}$. | Sensible a la selección del kernel y parámetros $C/\gamma$; no escala bien con grandes volúmenes de datos brutos. | 82% - 89% (según extractor)$^{11}$. |
| **Random Forest** | Exige extracción previa de características tabulares/estadísticas$^{4}$. | Inmune al sobreajuste leve; maneja relaciones no lineales; proporciona métricas de importancia de variables$^{4}$. | Ineficiente ante transformaciones de rotación o traslación complejas en píxeles brutos. | 84% - 90%$^{4}$. |
| **XGBoost** | Vectores de características cuantitativos estructurados$^{31}$. | Altísima velocidad de entrenamiento; optimización de gradiente robusta ante datos desbalanceados. | Requiere un diseño de características (*feature engineering*) manual sumamente riguroso. | 85% - 91%$^{31}$. |
| **CNN: ResNet-50** | Imágenes procesadas/rescaladas en formato matricial RGB ($256 \times 256$)$^{3}$. | Las conexiones residuales superan el desvanecimiento del gradiente; excelente capacidad de abstracción$^{32}$. | Elevado número de parámetros; requiere GPU para entrenamiento e inferencia veloz. | 91% - 95%$^{31}$. |
| **CNN: MobileNetV2/V3** | Imágenes procesadas/rescaladas en formato matricial RGB$^{31}$. | Arquitectura liviana (*depthwise separable convolutions*); ideal para despliegue en dispositivos *edge* y móviles$^{33}$. | Leve degradación de la exactitud frente a redes densas en escenarios de oclusión severa. | 89% - 93%$^{31}$. |

## 6. Matriz Multidimensional de Impactos Potenciales

El desarrollo e implementación de una solución automatizada de clasificación de productos frescos genera repercusiones sistémicas. En el marco de la fase de Comprensión del Negocio, se presenta a continuación la matriz multidimensional de impactos potenciales. Dado que el proyecto se encuentra en fase de desarrollo universitario, estos elementos se plantean como hipótesis de impacto orientadoras y no como resultados empíricos demostrados.

| **Dimensión de Impacto** | **Beneficios Potenciales Esperados** | **Riesgos e Ineficiencias Asociadas** | **Limitaciones Técnicas u Operativas** | **Estrategias de Mitigación Propuestas** |
| --- | --- | --- | --- | --- |
| **Económico** | Reducción del tiempo de inspección por unidad; disminución de devoluciones en *retail*; optimización del valor de venta mediante segregación de calidad. | Elevado costo inicial de infraestructura de captura; pérdidas económicas por falsos positivos/negativos del modelo. | Imposibilidad de evaluar calidad interna (°Brix, firmeza) mediante visión RGB convencional$^{10}$. | Diseñar el sistema como una herramienta *edge* accesible vía teléfonos móviles; integrar validación humana en casos de incertidumbre. |
| **Social** | Erradicación de tareas de inspección manual penosas y repetitivas; estandarización justa del pago a pequeños agricultores. | Desplazamiento laboral de clasificadores manuales no cualificados; brecha de acceso tecnológico para pequeños campesinos. | Resistencia al cambio operativo en plazas de mercado tradicionales por desconfianza en algoritmos. | Reclasificar el rol laboral humano hacia la supervisión del sistema y empaque técnico; interfaces móviles intuitivas de uso libre. |
| **Ambiental** | Reducción de desperdicios biológicos poscosecha mediante segregación temprana; canalización de fruta "fea" a agroindustria$^{1}$. | Incremento de la huella de carbono asociada al entrenamiento de modelos computacionales y desecho de hardware (*e-waste*). | El sobre-descarte estético por algoritmos estrictos podría incrementar irónicamente el desperdicio de alimento comestible. | Ajustar los umbrales de decisión para categorizar producto imperfecto pero comestible hacia mercados secundarios o caridad. |
| **Global** | Alineación con los Objetivos de Desarrollo Sostenible (ODS 12.3: reducción de pérdidas de alimentos); facilitación del comercio internacional$^{1}$. | Creación de barreras no arancelarias implícitas si las cadenas globales exigen estándares digitales inaccesibles para el sur global. | Dependencia de arquitecturas de software propietarias y conjuntos de datos sesgados hacia variedades de países desarrollados$^{3}$. | Promover el uso de conjuntos de datos locales abiertos y marcos de aprendizaje automático de código abierto (*open-source*). |

## 7. Dilemas Éticos, Sesgos y Estrategias de Mitigación

### 7.1 Impacto Socio-Laboral y Desplazamiento de Mano de Obra

La automatización de procesos de selección visual en plantas de acopio de menor escala o cooperativas agrícolas plantea un dilema ético directo respecto al impacto en el empleo rural. La inspección manual es una fuente primaria de trabajo informal para comunidades vulnerables. Reemplazar completamente al personal de selección sin un plan de transición puede exacerbar la precariedad socioeconómica local.

*Mitigación*: Fomentar un esquema de **Humano en el Bucle (*****Human-in-the-Loop*****)**, donde el sistema automatizado asume el filtrado masivo de alta velocidad, y los trabajadores son capacitados para auditar las decisiones bordes del modelo, supervisar la logística de empaque y gestionar el mantenimiento básico del sistema.

### 7.2 Sesgos por Variedad Geográfica y Cambio de Dominio (***Domain Shift***)

Un sistema entrenado exclusivamente con imágenes de frutas producidas en una región geográfica determinada (e.g., India, de donde proviene el dataset *FruitNet*) desarrollará un sesgo inherente hacia la morfología, pigmentación y patrones de imperfección cromática propios de esos cultivares$^{3}$. Al desplegar dicho modelo en el contexto agroalimentario colombiano (e.g., plazas de mercado del Valle del Cauca), el rendimiento del algoritmo decaerá sensiblemente. Variedades locales de cítricos o plátanos presentan tonalidades y rugosidades epidérmicas que el modelo indio clasificaría erróneamente como "deterioro" o "enfermedad".

*Mitigación*: Aplicar técnicas de **adaptación de dominio (*****domain adaptation*****)** y enriquecer obligatoriamente el conjunto de datos de entrenamiento mediante la incorporación de fotografías propias tomadas en entornos locales bajo iluminación natural y artificial heterogénea.

### 7.3 Consecuencias Transaccionales de los Errores de Clasificación

Los errores de predicción del modelo conllevan asimetrías de daño económico:

- **Falso Positivo (Clasificar un fruto podrido como "Bueno")**: Provoca la rotura de la cadena de frío o la contaminación fúngica en cadena dentro del empaque durante el transporte, generando el rechazo total del lote en destino final$^{8}$.
- **Falso Negativo (Clasificar un fruto sano como "Malo")**: Perjudica directamente la liquidez del agricultor, al degradar injustamente el valor comercial de su cosecha.

*Mitigación*: Implementar la **estimación de incertidumbre** en las predicciones del modelo (mediante salidas *softmax* calibradas o redes bayesianas). Cuando la probabilidad asignada a una clase caiga dentro de una franja de duda (e.g., entre  0.40 y 0.65), el sistema debe marcar el ejemplar para una revisión visual humana secundaria.

## 8. Análisis Técnico del Conjunto de Datos y Brecha de Generalización

### 8.1 Origen y Estructura del Dataset de Referencia (***FruitNet / Kaggle***)

El dataset principal seleccionado como referencia para este proyecto es *Fruit Quality Classification*, alojado en la plataforma Kaggle por Ryan Park$^{6}$. El análisis cruzado de la literatura científica confirma que este repositorio proviene directamente del trabajo científico titulado *"FruitNet: Indian fruits image dataset with quality for machine learning applications"*, publicado por Vishal Meshram y Kailas Patil (2022) en la revista *Data in Brief* (DOI: 10.1016/j.dib.2021.107686)$^{3}$.

El dataset *FruitNet* fue construido para responder a la escasez de conjuntos de datos abiertos de frutas que incluyesen especímenes tanto sanos como defectuosos en entornos no estructurados$^{3}$.

#### Especificaciones Técnicas Confirmadas de FruitNet

- **Volumen Total de Imágenes**: 19,526 imágenes en formato .jpg$^{3}$.
- **Especies Incluidas**: 6 frutas de alto consumo y exportación: Manzana (*Apple*), Banano (*Banana*), Guayaba (*Guava*), Lima/Limón (*Lime*), Naranja (*Orange*) y Granada (*Pomegranate*)$^{3}$.
- **Categorías de Calidad**:
    1. Good Quality Fruits (11,664 imágenes): Frutos sanos, sin defectos aparentes$^{3}$.
    1. Bad Quality Fruits (6,778 imágenes): Frutos con podredumbre, necrosis, magulladuras severas o ataques de plagas$^{3}$.
    1. Mixed Quality Fruits (1,074 imágenes): Escenas que contienen simultáneamente especímenes sanos y deteriorados$^{3}$.
- **Adquisición**: Fotografía mediante cámaras traseras de tres dispositivos móviles (Apple iPhone 6 de 8 MP, ZUK Z2 Plus de 13 MP, Realme 5 Pro de 48 MP) en condiciones de luz natural y artificial, variando ángulos (frontal, superior, posterior, inferior, rotado 180°) y fondos (superficies oscuras, césped, suelo, fondos claros, multicolores)$^{3}$.
- **Dimensiones y Preprocesamiento**: Las imágenes originales de resolución $3024 \times 3024$ fueron reescaladas mediante un script de Python a una resolución estandarizada de $256 \times 256$  píxeles para las carpetas de buena y mala calidad, y $256 \times 192$ píxeles para la carpeta de calidad mixta, con una profundidad de color de 24 bits sRGB y resoluciones horizontales/verticales de 72 a 96 dpi$^{3}$.

| **Clase de Fruta** | **Calidad Buena (Good)** | **Calidad Mala (Bad)** | **Calidad Mixta (Mixed)** | **Total por Fruta** |
| --- | --- | --- | --- | --- |
| **Manzana (Apple)** | 1,149 | 1,141 | 113 | 2,403 |
| **Banano (Banana)** | 1,113 | 1,087 | 285 | 2,485 |
| **Guayaba (Guava)** | 1,152 | 1,129 | 148 | 2,429 |
| **Lima (Lime)** | 1,094 | 1,085 | 278 | 2,457 |
| **Naranja (Orange)** | 1,216 | 1,159 | 125 | 2,500 |
| **Granada (Pomegranate)** | 5,940 | 1,187 | 125 | 7,252 |
| **Total por Calidad** | **11,664** | **6,778** | **1,074** | **19,526** |

### 8.2 Evaluación de Capacidades Técnicas del Dataset Frente a los Objetivos del Proyecto

#### 1. Clasificación de Calidad Visual: **Factible**

El dataset proporciona un volumen robusto de datos (19,526 imágenes) ampliamente suficiente para entrenar redes convolucionales (CNN) y clasificadores tradicionales que aprendan a discriminar características de degradación epidérmica, podredumbre expuesta y cambios cromáticos severos$^{2}$.

#### 2. Estimación de Tamaño y Calibre: **Infactible con el Dataset Nativo**

Existe una limitación metodológica crítica en el dataset *FruitNet*: **No permite entrenar de forma directa un modelo de estimación de tamaño absoluto o diámetro físico normalizado (en mm)**.

Esta imposibilidad se fundamenta en tres razones técnicas:

1. **Pérdida de Escala Espacial por Reescalado**: Todas las imágenes fueron redimensionadas por software a matrices fijas de $256 \times 256$ píxeles$^{3}$. Este proceso altera la relación aspecto-distancia y destruye la escala espacial nativa.
1. **Ausencia de Marcadores Fiduciarios o Referencias de Escala**: Las fotografías no incluyen un objeto de dimensiones conocidas (e.g., una regla graduada, una moneda o un patrón calibrado) en el plano de la escena$^{3}$.
1. **Variabilidad Indeterminada de la Distancia Focal y de Captura**: Al ser capturadas a mano alzada con tres teléfonos móviles distintos a distancias no estandarizadas, un fruto pequeño fotografiado de cerca ocupa exactamente el mismo número de píxeles en la matriz de $256 \times 256$ que un fruto grande fotografiado a mayor distancia$^{3}$.

*Solución para el Proyecto*: La estimación de tamaño dentro del proyecto universitario únicamente se podrá ejecutar mediante la recolección del conjunto de datos complementario de 30 a 50 fotografías locales. Estas tomas deben realizarse obligatoriamente utilizando una **distancia de cámara a la mesa fija** y/o un **marcador fiduciario de escala graduada** en el fondo, lo cual permitirá computar el factor de conversión de píxeles a milímetros reales ($\frac{mm}{pixel}$).

### 8.3 Brecha de Generalización con el Entorno Agroalimentario Colombiano

Al intentar desplegar un modelo entrenado sobre *FruitNet* en los mercados o centrales de acopio de Colombia (e.g., CAVASA o supermercados del Valle del Cauca), surgirán discrepancias de distribución de datos (*data drift*):

- **Divergencia Varietal**: La "Lima" del dataset indio (*Citrus aurantifolia*) diffiere morfológica y cromáticamente del "Limón Tahití" (*Citrus latifolia*) predominantemente comercializado y exportado en Colombia. De igual manera, las variedades de banano, guayaba y manzana india difieren en morfología y manchado natural con respecto a los cultivares locales (e.g., Plátano Hartón, Banano Criollo, Guayaba Manzana).
  
- **Condiciones de Iluminación y Contenedores**: En las plazas colombianas, las frutas se presentan amontonadas en canastillas plásticas de color naranja o azul (NTC 5166), sobre sacos de fique o en mostradores de madera con iluminación incandescente o sombreado al aire libre, escenarios no representados cabalmente en el dataset de origen$^{3}$.

## 9. Oportunidad de Negocio y Justificación del Proyecto

A pesar de las limitaciones inherentes a los datos de acceso público, la concepción de un sistema inteligente de inspección visual representa una oportunidad técnica y comercial altamente relevante en el ámbito de la Ingeniería de Sistemas aplicada a la agroindustria.

### 9.1 Propuesta de Valor Técnico-Comercial

La solución no pretende sustituir las complejas líneas industriales de calibrado hiperespectral de gran escala, cuyo costo de instalación resulta inalcanzable para pequeños acopiadores. La propuesta de valor radica en un **sistema liviano de bajo costo (*****Edge AI*****)** ejecutable en dispositivos móviles o computadores convencionales equipados con cámaras web de alta definición.

Este sistema asistirá a pequeños productores, intermediarios en plazas de mercado (e.g., CAVASA) y secciones de recepción de supermercados locales en las siguientes tareas:

1. **Pre-segregación Automatizada de Velocidad**: Separar rápidamente los productos que presentan daños epidérmicos o podredumbre visible (Bad Quality), impidiendo su almacenamiento junto a especímenes sanos$^{3}$.
1. **Estandarización del Juzgamiento Visual**: Proporcionar un criterio cuantitativo e imparcial de color y textura que reduzca las discusiones transaccionales sobre la calidad del lote entregado$^{2}$.
1. **Calibrado Relativo Asistido por Escala**: Medir el diámetro de los especímenes mediante el uso de marcadores fiduciarios simples en la mesa de captura local, permitiendo categorizar el producto en calibres pequeño, mediano y grande conforme a rangos ajustables$^{25}$.

## 10. Conclusiones para la Fase de Comprensión del Negocio (CRISP-DM)

La realización de esta investigación exhaustiva permite consolidar los hallazgos fundamentales requeridos para formalizar la fase de **Comprensión del Negocio (*****Business Understanding*****)** dentro del proyecto:

1. **Relevancia del Problema Sectorial**: La inspección manual hortofrutícola es un cuello de botella crítico que contribuye directamente a las pérdidas poscosecha (39% de pérdidas en frutas y verduras en Colombia, equivalentes a 1.42 millones de toneladas)$^{1}$. La automatización visual ataca un problema real de ineficiencia económica y desperdicio de alimento.

2. **Delimitación del Alcance Espectral (RGB)**: La tecnología basada en fotografías RGB estándar permite evaluar objetivamente la **calidad visual**, el **estado de madurez cromático externo** y la **presencia de defectos epidérmicos o deterioro expuesto**$^{4}$. No debe promocionarse como un método para medir parámetros internos organolépticos (°Brix, acidez titulable) o nutricionales, los cuales requieren sensores hiperespectrales o pruebas químicas NIRS$^{10}$.

3. **Claridad en la Taxonomía de Clases**: Las etiquetas del modelo (Good, Bad, Mixed) constituyen pre-clasificaciones operativas visuales y no equivalen automáticamente a los grados comerciales reglamentados (*Extra*, *Primera*, *Segunda*) por normas como la NTC 1291, NTC 4103, UNECE o Codex Alimentarius$^{8}$. El sistema debe actuar como una herramienta de apoyo a la segregación y no como un certificador legal.

4. **Diagnóstico del Dataset Principal (*****FruitNet / Kaggle*****)**: El conjunto de datos de 19,526 imágenes es técnicamente apto y robusto para entrenar clasificadores de calidad visual (CNN y ML tradicional)$^{2}$. Sin embargo, **es nulo para estimar el tamaño o calibre real** debido al reescalado sintético a $256 \times 256$ y a la ausencia de marcas de escala espacial$^{3}$.

5. **Estrategia para el Tamaño y Adaptación Local**: La estimación de tamaño y la superación del sesgo de distribución geográfica (*domain shift*) dependen críticamente de la correcta recolección del dataset local complementario (30 a 50 imágenes de frutas colombianas). Este subconjunto debe ser fotografiado bajo un protocolo riguroso con cámara a distancia fija y uso de marcadores fiduciarios de escala física ($\frac{mm}{pixel}$).

6. **Factibilidad de Ingeniería de Sistemas**: El proyecto es técnicamente viable combinando arquitecturas CNN ligeras (e.g., MobileNetV3 o ResNet-50 mediante transfer learning) para el procesamiento en tiempo real$^{31}$, proporcionando un prototipo de alto valor para la modernización de la gestión agrícola regional en el Valle del Cauca.

## Fuentes citadas

1. [PÉRDIDA Y DESPERDICIO DE ALIMENTOS EN COLOMBIA - DNP](https://colaboracion.dnp.gov.co/CDT/Prensa/Publicaciones/P%C3%A9rdida%20y%20desperdicio%20de%20alimentos%20en%20colombia.pdf)

2. [Fruit Quality Classification Using AI: A Machine Learning-Based](https://ijsret.com/wp-content/uploads/2025/03/IJSRET_V11_issue2_362.pdf)

3. `1-s2.0-S2352340921009616-main.pdf`

4. [Object Detection and Recognition Techniques Based on Digital](https://www.mdpi.com/2073-4395/13/3/639)

5. [Fruit Detection and Recognition Based on Deep Learning for ... - MDPI](https://www.mdpi.com/2073-4395/13/6/1625)

6. [Fruit Quality Classification - Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)

7. [NTC 1291: Normas Frutas y Hortalizas | PDF - Scribd](https://es.scribd.com/doc/141473952/NORMA-TECNICA-NTCCOLOMBIANA-1291)

8. [Frutas y Hortalizas Frescas](https://www.fao.org/4/a1389s/a1389s00.pdf)

9. [Fresh Truffle Grades Explained: EXTRA, A, B and C - Terra Ross](https://www.terra-ross.com/blogs/truffle-guides/fresh-truffle-grades-extra-a-b-c)

10. [(PDF) Non-Destructive Detection of Soluble Solids Content in Fruits](https://www.researchgate.net/publication/394745974_Non-Destructive_Detection_of_Soluble_Solids_Content_in_Fruits_A_Review)

11. [Image Acquisition, Preprocessing and Classification of Citrus Fruit](https://www.researchgate.net/publication/371665326_Image_Acquisition_Preprocessing_and_Classification_of_Citrus_Fruit_Diseases_A_Systematic_Literature_Review)

12. [NTC 5166 | kontii](https://kontii.files.wordpress.com/2012/10/ntc-5166.pdf)

13. [MINISTERIO DE AGRICULTURA Y DESARROLLO RURAL](https://www.minagricultura.gov.co/fileadmin/normatividad/proyectos-normativos/Proyecto_Normativo_Disminucion_de_las_perdidas_y_los_desperdicios_de_alimentos.pdf)

14. [ANÁLISIS SITUACIONAL - UPRA](https://upra.gov.co/sites/default/files/2026-05/01_DT_A_Situacional.pdf)

15. [Priorización de alternativas productivas agropecuarias y análisis de](https://upra.gov.co/sites/default/files/2026-11/11_PriorizNteSantander20241108.pdf)

16. [PRODUCTIVIDAD Y COMPETITIVIDAD FRUTICOLA ANDINA](https://old.fontagro.org/new/uploads/productos/16111_-_Producto_9.pdf)

17. [Análisis de potencialidades productivas de mujeres](https://colombia.unwomen.org/sites/default/files/2023-09/analisis_de_potencialidades_productivas_de_mujeres.ai_compressed_1.pdf)

18. [Plan integral de desarrollo agropecuario y rural con enfoque territorial](https://www.adr.gov.co/wp-content/uploads/2021/07/TOLIMA-TOMO-II.pdf)

19. [NTC 1291: Clasificación de Frutas Frescas | PDF | Naturaleza - Scribd](https://es.scribd.com/doc/58310079/NTC-1291-Fruta-y-Hortalizas-Generalidades)

20. [NORMA TÉCNICA COLOMBIANA NTC 4103 FRUTAS FRESCAS](https://pdfcoffee.com/download/norma-tecnica-colombiana-ntc-4103-para-fresa-4-pdf-free.html)

21. [Ntc4103 Fresa - Calaméo](https://www.calameo.com/books/004339121cb3d29d9e010)

22. [Produce RFQ Checklist for B2B Buyers | DeltaMasr Fresh](https://deltamasrfresh.com/en/knowledge/produce-rfq-checklist-buyer-guide/)

23. [Especificaciones NTC 4103 Fresa | PDF | Fruta - Scribd](https://es.scribd.com/doc/110232865/Norma-Tecnica-Colombiana-Ntc-4103-Para-Fresa)

24. [Object Detection and Recognition Techniques Based on Digital](https://www.researchgate.net/publication/368765952_Object_Detection_and_Recognition_Techniques_Based_on_Digital_Image_Processing_and_Traditional_Machine_Learning_for_Fruit_and_Vegetable_Harvesting_Robots_An_Overview_and_Review)

25. [Tabla de calibre de la fruta: guía de selección y clasificación - Futura](https://www.futura-grading.com/es/tabla-de-calibre-de-la-fruta-guia-de-seleccion-y-clasificacion/)

26. [UNECE STANDARD FFV-50 APPLES](https://unece.org/sites/default/files/2020-12/50_Apples.pdf)

27. [Lemon quality guide and export standards - Wikifarmer](https://wikifarmer.com/library/en/article/lemon-quality-guide-and-export-standards)

28. [UNECE STANDARD FFV-35 STRAWBERRIES](https://unece.org/sites/default/files/2023-12/FFV35_Strawberries_2023_e.pdf)

29. [Técnicas postcosecha para productos hortofrutícolas](https://sired.udenar.edu.co/14764/1/Cartilla_postcosecha.pdf)

30. [Norma técnica NTC colombiana 5422 - Bogotá - Corabastos](https://www.corabastos.com.co/sites/default/files/2020-08/NormaTecnicaColombianaNTC5422.pdf)

31. [Computer Vision and Machine Learning in Agriculture, Volume 3](https://dokumen.pub/computer-vision-and-machine-learning-in-agriculture-volume-3-algorithms-for-intelligent-systems-9819937531-9789819937530.html)

32. [Smart Agriculture Applications Using Deep Learning Technologies](https://www.mdpi.com/2076-3417/12/12/5919)

33. [(PDF) Fruit Detection and Recognition Based on Deep Learning for](https://www.researchgate.net/publication/371715666_Fruit_Detection_and_Recognition_Based_on_Deep_Learning_for_Automatic_Harvesting_An_Overview_and_Review)

34. [Automatic date fruit sorting system based on machine learning and](http://thesis.univ-biskra.dz/6588/1/Doctorat_SOUTENANCE_UN0701__DOC_DEMANANDE_290122%20%281%29%20%5B1%5D_merge.pdf)

35. [(PDF) FruitNet: Indian fruits image dataset with quality for machine](https://www.researchgate.net/publication/356846683_FruitNet_Indian_fruits_image_dataset_with_quality_for_machine_learning_applications)

36. [FruitNet: Indian fruits image dataset with quality for machine learning](https://pubmed.ncbi.nlm.nih.gov/34917715/)

37. [FruitNet: Indian Fruits Dataset with quality (Good, Bad & Mixed)](https://data.mendeley.com/datasets/b6fftwbr2v/3)
