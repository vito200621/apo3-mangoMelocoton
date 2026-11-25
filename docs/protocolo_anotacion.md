# Protocolo de anotación de datos (borrador para aprobación del equipo)

Este protocolo es una propuesta operativa para revisar y aprobar en equipo antes de etiquetar. No sustituye una revisión humana ni constituye una etiqueta oficial hasta que se aprueben los criterios y se resuelva una muestra piloto.

## Unidad y trazabilidad

- Una fila representa un producto identificable en una imagen propia o un recorte Mixed. Si hay más de un producto en una foto, anotar cada recorte por separado y conservar el mismo `group_id` de la escena original.
- No etiquetar una escena completa como si fuera un solo producto. Descartar recortes sin un producto identificable o con varios productos que no se puedan separar.
- No inferir el tamaño físico a partir de píxeles. La clase de calidad y la madurez son campos distintos.
- Conservar `sample_id`, `source`, `source_image_id`, `path` y `group_id` generados por el pipeline. No editar identificadores ni rutas.

## Calidad visible: criterios propuestos

Evaluar defectos observables en el producto, no el fondo, la iluminación ni el gusto personal. Usar estas definiciones para el piloto y acordar umbrales por fruta si la apariencia normal varía mucho.

| Clase | Criterio observable propuesto | Ejemplos |
| --- | --- | --- |
| `Good` | Producto comercializable, sin deterioro visible; se toleran marcas cosméticas pequeñas que no comprometen la pulpa. | Coloración relativamente uniforme para su madurez; sin moho, pudrición, cortes profundos o golpes importantes. |
| `Regular` | Producto aún utilizable/comercializable con defecto o madurez no ideal, pero sin deterioro severo. | Magulladura leve, mancha/cicatriz localizada, madurez irregular o daño cosmético moderado. |
| `Bad` | Deterioro claro o defecto severo que lo hace no comercializable para consumo fresco. | Moho, pudrición, fuga/ablandamiento severo, corte profundo o daño extenso. |

Registrar el defecto en `defect` con términos breves y consistentes (por ejemplo, `bruise`, `cut`, `mold`, `rot`, `discoloration`, `wrinkling`); usar `none` si no se observa. Si hay varios, separarlos con `;`. No convertir automáticamente madurez avanzada en `Bad`: el equipo debe distinguir sobremadurez sin pudrición de deterioro.

## Madurez

Usar `maturity` independientemente de calidad: `unripe`, `turning`, `ripe`, `overripe` o `unknown`. Acordar referencias visuales por fruta (color/textura) antes de etiquetar todo; una misma clase de calidad puede contener distintos estados de madurez. No inventar un estado si la especie o la imagen no permiten juzgarlo.

## Tamaño y medida de referencia

- En `diameter_mm`, registrar el diámetro transversal máximo medido con regla o calibre, en milímetros; anotar el método de medición en `notes`. Mantener la unidad y el instrumento consistentes.
- Asignar `size_label` (`small`, `medium`, `large`) solo con límites previamente acordados por especie o variedad. No usar umbrales universales entre frutas distintas. Si faltan límites o una medida fiable, usar `unknown`.
- `diameter_norm` es una estimación relativa a la imagen. No equivale a milímetros y no es una referencia física.

## Contexto de captura y privacidad

Para fotos propias completar `fruit`, `capture_device`, `lighting`, `background`, `source`, calidad, madurez y medida disponible. Registrar condiciones en categorías concisas (por ejemplo, `indoor diffuse`, `outdoor shade`; fondo `plain white`, `wood`, `market`). No incluir nombres, ubicaciones exactas, números de teléfono ni otros datos personales.

Obtener permiso del propietario/fotógrafo y de las personas responsables del lugar cuando corresponda. Evitar rostros, matrículas, documentos, pantallas y objetos que identifiquen hogares o personas. Eliminar EXIF/GPS antes de compartir o versionar las imágenes; los datos originales y procesados permanecen fuera de Git según `.gitignore`.

## Revisión y resolución de desacuerdos

1. Dos integrantes etiquetan independientemente una muestra piloto estratificada por fruta y calidad esperada.
2. Comparar desacuerdos en calidad, defecto, madurez y tamaño. Revisar juntos los casos con la imagen original y aplicar los criterios escritos; actualizar el protocolo antes de continuar.
3. Para el resto, asignar una persona anotadora y una revisora para los casos dudosos y una muestra aleatoria de cada clase/fuente. Si no hay acuerdo, dejar el registro pendiente de adjudicación; no forzar una etiqueta.
4. Antes de entrenar, verificar que no haya etiquetas vacías/ajenas, identificadores duplicados, rutas rotas ni grupos compartidos entre train/validation/test.

## Flujo de archivos

1. Guardar fotos propias bajo `data/raw/own_collection/<Good|Regular|Bad>/<fruit>/`; conservar nombres de archivo sin información personal. El pipeline crea `data/annotations/own_collection_annotations.csv` como plantilla.
2. Ejecutar el cuaderno 02 para generar recortes de escenas Mixed y la plantilla `data/annotations/mixed_candidates_annotations.csv`.
3. Completar la plantilla de recortes revisados. Calidad debe ser exactamente `Good`, `Regular` o `Bad`; filas vacías siguen pendientes y se excluyen.
4. Volver a ejecutar el cuaderno 02 con reinicio de salidas procesadas. Las etiquetas se unen por `sample_id`, pero las rutas, procedencia y `group_id` siempre se toman del manifiesto generado, no de campos editables de la plantilla.
5. Revisar conteos por calidad/fruta/fuente y el resumen de particiones antes de entrenar.

El entrenamiento utiliza registros con etiquetas revisadas y trazables; las medidas físicas se reportan solo cuando fueron tomadas y verificadas.
