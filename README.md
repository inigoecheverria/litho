# Código Termal 0.0.1

-   [Prerequisitos](#prerequisitos)
    -   [Anaconda (interfaz gráfica)](#anaconda-interfaz-gráfica)
    -   [Miniconda (uso en la terminal)](#miniconda-uso-en-la-terminal)
-   [Uso del código termal](#uso-del-código-termal)
    -   [Desde Jupyter notebook](#desde-jupyter-notebook)
    -   [Desde un script](#desde-un-script)
    -   [Ejecutar un script disponible](#ejecutar-un-script-disponible)

## Prerequisitos

Anacona o Miniconda son opciones recomendadas para instalar los
requerimientos de este código. En Windows trabajar con Anaconda es lo
más simple, para Mac o Linux se recomienda Miniconda.

### Anaconda (interfaz gráfica)

#### Descargar e instalar Anaconda

Descargar e instalar el [ejecutable](https://www.anaconda.com/products/individual#Downloads)


#### Instalar los paquetes de python necesarios

-   xarray
-   matplotlib
-   cartopy
-   plotly
-   kaleido

### Miniconda (uso en la terminal)

#### Descargar e instalar Miniconda

-   Descargar e instalar el [ejecutable](https://docs.conda.io/en/latest/miniconda.html)

-   Verificar instalación: Abrir una terminal, y ejecutar el comando
    `conda --version`, para confirmar que la instalación tuvo éxito.
    Luego ejecutar el comando `echo $PATH`, y comprobar que aparece
    "miniconda3" entre el output, si es así­ el programa se ha
    configurado correctamente.

-   Actualizar: Ejecutar el comando `conda update conda`, y responder
    con `y`.

#### Configurar un ambiente virtual:

La opcion más sencilla es ejecutar `conda env create -f env.yml` desde
la carpeta donde se ha descomprimido el código, para realizar una
configuración automática. La terminal debería preguntar por confirmación
un par de veces (siempre responder con `y`), y luego finalizar con un
mensaje de éxito.

Finalmente ejecutar `conda activate cod_termal` para activar el ambiente
virtual.

En caso contrario, seguir los siguientes dos pasos:

1.  Crear y activar: Ejecutar el comando
    `conda create --name cod_termal python=3.9.1` para crear un ambiente
    virtual. Luego ejecutar `conda activate cod_termal`, para activarlo.
    A la izquierda del prompt de la terminal ahora debería aparecer
    `(cod_termal)`.

2.  Instalar paquetes de python necesarios manualmente: Ejecutar los
    siguientes comandos en la terminal para instalar los respectivos
    paquetes (cada vez que se pida confirmación responder con `y`):

-   xarray:
    `conda install -c conda-forge xarray dask netCDF4 bottleneck`
-   matplotlib: `conda install matplotlib`
-   cartopy: `conda install -c conda-forge cartopy`
-   jupyter lab: `conda install -c conda-forge jupyterlab`
-   plotly: `conda install -c plotly plotly`
-   kaleido: `conda install -c plotly python-kaleido`

#### Tips

-   Se puede instalar la librería litho es mediante el comando
    `pip install -e .`
    Esto permite que python la use como paquete, y aun asi editarla.
-   Un método recomendado para ejecutar los scripts es
    `python -m scripts.vars_expected`

## Uso del código termal

### Desde Jupyter notebook

La forma más sencilla de utilizar el codigo es con jupyter notebook. En
la carpeta donde se descomprimió el código se encuentra el archivo
`Ejemplos.ipynb`, que sirve a base de introducción; se puede abrir desde
Anaconda o, desde una terminal ubicada en esa carpeta, ejecutando
`jupyter notebook Ejemplos.ipynb` o `jupyter lab` (el ambiente virtual
debe estar activado).

Esto abrirá automáticamente el navegador con el notebook listo para ser
editado. En caso de que aparezca una ventana emergente preguntando qué
kernel de python se desea usar, seleccionar "Python 3". En cualquier
caso, antes de seguir con los siguientes pasos, confirmar que en la
esquina superior derecha se mencione que Python 3 es el kernel de python
activo.

El archivo `Ejemplos.ipynb` contiene ejemplos de como utilizar el código
para obtener distintos tipos de resultados. Para ejecutar una celda y
ver su resultado presionar el boton ▶️ con la celda seleccionada (es
importante ejecutar las celdas de código en orden para que funcionen
correctamente). Para ejecutar todas las celdas en orden presionar el
botón ⏩.

El archivo puede ser editado sin preocupaciones, pero es recomendable
crear una copia de respaldo para tenerlo a mano.

### Desde un script

El mismo código que funciona en un notebook puede ser utilizado desde un
archivo (ej. `script.py`) para realizar tareas que no requieran output
interactivo.

### Ejecutar un script disponible

Se incluyen dos scripts:

#### vars_stats.py:

Uso:

`python vars_stats.py var1 '[start1, stop1, step1]' var2 '[start2, stop2, step2]' ...`

-   `var1`, `var2`, etc., deben ser reemplazados por los nombres de las
    variables que se pueden encontrar en `inputs.py`
-   `'[start, stop, step]'`, definen el rango de valores que la
    respectiva variable asumirá

Descripción:

Para cada combinación posible de diferentes variables en los rangos
definidos, este script ejecuta un modelo termal que utiliza esas
variables como input. El modelo termal que se ejecutará es el dado por
el valor de `thermal_conf['default_model']` que se encuentra definido en
`inputs.py`. Todas las variables que no sean especificadas asumirán los
valores previamente definidos en `inputs.py`. El output consiste en un
valor de RMSE y MSE para cada combinación ejecutada, los cuales son
calculados a partir de la comparación entre los valores del flujo de
calor superficial predichos por el modelo y los provenientes de
mediciones recopiladas en el archivo `litho/data/shf_data.dat` (modelo -
prediccion). Estos valores son almacenados en forma de texto y gráficos
en la carpeta `vars_stats`, situada dentro del directorio definido por
`thermal_conf['output_path']` en `inputs.py`, junto con una copia del
archivo `inputs.py` que se utilizó para generarlos. Para reconstruir los
graficos a partir de los archivos de texto ya generados se puede
ejecutar `python vars_stats.py` sin argumentos.

#### vars_expected.py:

Uso:

`python vars_expected.py`

Descripción:

Este script invierte las funciones usadas en los distintos modelos
termales para calcular el flujo de calor superficial (Q), de modo que
para cada uno de los registros de mediciones de este parámetro,
presentes en en el archivo `litho/data/shf_data.dat`, con formato
`[lon, lat, Q, error]`, este script calcula los valores medios, máximos
y mínimos que deberían tener las variables termales para que el flujo de
calor modelado en el punto (lon, lat) se encuentre dentro de los
margenes de error de la respectiva medición. Cada variable es calculada
individualmente asumiendo que los valores de las demas son los definidos
en `inputs.py`. En una ejecución se genera el output para todos los
modelos definidos. Los resutados son almacenados como archivos CSV en la
carpeta `vars_expected` contenida en el directorio especificado por
`thermal_conf['output_path']` en `inputs.py`.
