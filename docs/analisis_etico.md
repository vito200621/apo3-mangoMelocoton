# Análisis ético y de impactos

Este análisis es preliminar. Las conclusiones sobre impactos deben actualizarse con datos y resultados medidos; el prototipo no demuestra por sí solo reducción de pérdidas o desperdicio.

| Riesgo o impacto | Consecuencia posible | Mitigación y evidencia que se debe registrar |
| --- | --- | --- |
| Sesgo por fruta, variedad, fuente o condiciones de captura | El modelo puede acertar en las frutas/fondos de Kaggle y fallar en productos locales, iluminaciones o dispositivos distintos. | Contar soporte por clase × fruta × fuente; reportar métricas desglosadas y ejemplos de error; recopilar casos faltantes antes de extender el uso. No generalizar a frutas sin evaluación. |
| Etiquetas subjetivas | Umbrales diferentes entre anotadores producen errores de referencia y conclusiones engañosas. | Aplicar el protocolo de anotación, pilotear con dos anotadores y adjudicar desacuerdos; conservar defectos/madurez por separado. |
| Falso `Good` | Producto deteriorado podría avanzar en la cadena, con costos económicos o de confianza y posibles consecuencias de inocuidad. | La herramienta es apoyo, no certificación; conservar inspección humana y abstenerse/derivar cuando la imagen sea ilegible o la confianza insuficiente. No afirmar aptitud para consumo. |
| Falso `Bad` | Producto utilizable podría descartarse, generando pérdida económica y desperdicio. | Revisar falsos Bad, calibrar umbral de revisión y priorizar sensibilidad/recall por clase según el contexto real, sin ocultar trade-offs. |
| Fotografías propias y metadatos | Rostros, hogares, ubicaciones o identificadores pueden quedar en imágenes/EXIF. | Obtener consentimiento, evitar personas/elementos identificables, quitar EXIF/GPS, no guardar nombres personales en anotaciones y mantener fotos fuera de Git. |
| Automatización y trabajo | El uso sin revisión podría desplazar decisiones de trabajadores o excluir su conocimiento contextual. | Mantener decisión humana, explicar límites y diseñar la interfaz como asistencia; probarla con usuarios antes de cualquier operación real. |
| Coste computacional | La búsqueda de modelos y CNN consume tiempo, electricidad y almacenamiento. | Usar resolución y épocas justificadas, EarlyStopping, registrar duración/artefactos y evitar ejecuciones repetidas de test. |
| Impacto ambiental afirmado sin medir | Atribuir reducción de desperdicio sin seguimiento puede exagerar el beneficio. | No afirmar reducciones; medir en un piloto de campo masa descartada, errores y línea base antes/después, incluyendo intervalos y limitaciones. |
| Tamaño en píxeles confundido con tamaño físico | Una decisión comercial basada solo en encuadre/distancia puede ser incorrecta. | Mostrar “tamaño relativo”, no mm; calibrar con regla/calibre y validar independientemente antes de decidir precios o clasificación física. |

## Uso responsable

Los datos deben tener procedencia autorizada y propósito limitado a la evaluación académica. Los archivos originales y artefactos derivados no se publican sin revisar licencias, permisos y privacidad. Las clases con soporte ausente deben declararse como no evaluadas; una salida softmax o una predicción de la interfaz no equivale a evidencia de confiabilidad. Antes de despliegue fuera del curso se requiere validación externa, revisión humana, pruebas de errores y un procedimiento para retirar o corregir etiquetas/modelos.