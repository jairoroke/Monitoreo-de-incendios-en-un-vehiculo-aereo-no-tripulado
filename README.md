# Simulación del cuadrirrotor (ROS 2 Jazzy + Gazebo Harmonic)

Monitoreo de incendios forestales con un vehículo aéreo no tripulado: control super-twisting
adaptativo con discretización implícita, detección del fuego con YOLO26 y búsqueda autónoma.

Autor: Dr. Jairo Olguín Roque (FES Acatlán, UNAM).

Libro de la simulación, explicado desde cero (drones, control, visión, autonomía, instalación y uso):
[`src/cuadrirotor_sim/doc/manual/libro_drones_control_vision.pdf`](src/cuadrirotor_sim/doc/manual/libro_drones_control_vision.pdf)
(fuentes LaTeX en la misma carpeta).

## Clonar

El repositorio tiene la forma de un espacio de trabajo de ROS 2 (Ubuntu 24.04, ROS 2 Jazzy,
Gazebo Harmonic):

    git clone https://github.com/jairoroke/Monitoreo-de-incendios-en-un-vehiculo-aereo-no-tripulado.git ~/ros2_ws
    git clone https://github.com/ArduPilot/ardupilot_gazebo.git ~/ros2_ws/src/ardupilot_gazebo   # solo para controlador:=ardupilot

ArduCopter 4.7.2 SITL va en `~/ardupilot`; el detector YOLO usa el entorno `~/.venvs/yolo`.

## Compilar

    cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash

## Lanzar

    ros2 launch cuadrirotor_sim simulacion.launch.py                       # despegue y hover a 3 m
    ros2 launch cuadrirotor_sim simulacion.launch.py mision:=orbita        # órbita alrededor del foco (8, 5)
    ros2 launch cuadrirotor_sim simulacion.launch.py mision:=orbita viento:=true csv:=$HOME/corrida.csv
    ros2 launch cuadrirotor_sim simulacion.launch.py gui:=false Ts:=0.01   # sin interfaz, otro periodo de muestreo

    ros2 launch cuadrirotor_sim simulacion.launch.py controlador:=sta esquema:=implicito mision:=orbita viento:=true

    ros2 launch cuadrirotor_sim simulacion.launch.py mundo:=bosque controlador:=sta mision:=orbita

    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque controlador:=sta mision:=orbita

    # Valle boscoso de 420 x 420 m con montañas: el dron barre todo el bosque hasta encontrar el incendio
    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=sta mision:=busqueda
    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=sta mision:=busqueda foco_x:=-150 foco_y:=60
    # La misma búsqueda pilotada por ArduPilot SITL (abre las ventanas del SITL)
    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=ardupilot mision:=busqueda

Argumentos: `vehiculo` (cuadrirotor|tarot650), `mundo` (incendio|bosque|bosque_grande), `gui`, `controlador` (pd|sta|ardupilot|ninguno), `esquema` (explicito|semi_implicito|implicito),
`mision` (hover|orbita|busqueda), `Ts` [s], `viento` (true|false), `csv` (ruta), `foco_x`, `foco_y`
(posición del incendio en `bosque_grande`).

## Controlador super-twisting adaptativo (`controlador_sta`)

Por eje, s = e' + λe con e = p_d − p; la aceleración comandada es a = a_d + λe' − u, de modo
que s' = u + d. `sta.py` implementa u con ganancia adaptativa (Shtessel–Taleb–Plestan 2012)
y tres realizaciones discretas: `explicito`, `semi_implicito` e `implicito` (Brogliato, Polyakov y Efimov, 2020). El lazo de actitud es el mismo PD sobre SO(3) de la línea base.
Parámetros en `config/parametros.yaml`, sección `controlador_sta`.

Ojo: los `-p clave:=valor` de la línea de comandos **no** sobrescriben lo que esté en la
sección `controlador_sta:` del YAML (el YAML específico del nodo tiene prioridad). Para
campañas usa `scripts/corrida.sh`, que genera un YAML de ajustes:

    src/cuadrirotor_sim/scripts/corrida.sh ~/corridas/imp.csv 50 esquema=implicito Ts=0.02 k1_0=4.0 adaptativo=false
    src/cuadrirotor_sim/scripts/metricas.py ~/corridas/*.csv

## Resultados de verificación (21 oct 2026, órbita R = 4 m, viento 1 ± 0.5 N)

| Controlador | Ts | Error RMS con viento | Error máx. | Chattering |
|---|---|---|---|---|
| PD en cascada | 5 ms | 17.7 cm | 28.5 cm | 0.04 |
| STA adaptativo explícito | 5 ms | 2.4 cm | 4.7 cm | 0.4 |
| STA adaptativo semi-implícito | 5 ms | 2.1 cm | 4.0 cm | 0.5 |
| STA adaptativo implícito | 5 ms | 2.1 cm | 4.3 cm | 0.4 |
| STA explícito, k1 = 4 fijo | 20 ms | 2.8 cm (sin viento) | — | banda \|s_z\| p95 = 0.056 |
| STA implícito, k1 = 4 fijo | 20 ms | 1.7 cm (sin viento) | — | banda \|s_z\| p95 = 0.016 |

Observaciones relevantes:

1. Con perturbación, el esquema implícito de las ec. (9)–(12) deja una banda |s| ≈ Ts·|d|,
   **independiente de la ganancia** (insensible a la sobreestimación), pero no ∝ Ts²·k2.
   La predicción nominal s~ = s_k + Ts·u_k supone d = 0. Con `estimar_perturbacion: true`
   (d^ del paso anterior, Acary–Brogliato–Orlov 2012) la banda baja a ≈ Ts²·max|d'|.
   Ver `test/test_sta.py`.
2. En x, y el retardo del lazo de actitud y de los motores induce un ciclo límite
   (|s| ≈ 0.05–0.1 m/s) que crece con k1. Si el umbral de adaptación μ queda por debajo
   de él, la ganancia crece sin control (ganancia → máx., inestable). Por eso μ es por eje.
3. La velocidad que publica Gazebo es un promedio de 10 muestras (≈ 4.5 ms de retraso a
   1 kHz); es parte de la dinámica no modelada.

## STA en el lazo de actitud y esquema compensado (24 oct 2026)

Con `actitud:=sta` el lazo interno también es un STA: por eje del cuerpo,
s_R = −(ω + λ_R e_R) y τ = −J(λ_R ω + u_R), de modo que s_R' = u_R + d_R. Parámetros
`esquema_actitud`, `lambda_actitud`, `k1_actitud`, `adaptativo_actitud`, `mu_actitud`,
`tau_motor` y `horizonte_actitud` (sección `controlador_sta`).

El esquema `compensado` (en `sta.py`) mete la dinámica de primer orden del motor
(τ_m) en la inclusión implícita. Con u_k retenido durante H = N·Ts, el promedio de u_m es
g·u_k + (1 − g)·u_m,k, con g = 1 − (τ_m/H)(1 − e^(−H/τ_m)), y la inclusión conserva la forma
s~ + H k1 |s~|^½ ξ + H² k2 ξ = s_k + H ν_k: la solución cerrada es la misma. El modelo u_m se
corrige con el par que de verdad producen los rotores ya saturados (`Mezclador.realizado`).

Por qué: en x, y el error con el PD de actitud lo dominaba el ancho de banda del lazo
interno (ω_n ≈ 10 rad/s en el Tarot), no la discretización (`prototipos/eje.py`). En
actitud, con el motor de 40 ms, el implícito deja de ser insensible a la ganancia porque el
retardo también afecta al término λ_R ω del par (`test/test_sta.py`). El actuador
identificado en Gazebo es primer orden de 40 ms más una muestra de retardo puro.

Campaña (`scripts/campana_actitud.sh`, `scripts/resumen_actitud.py`): Tarot 650, órbita
con viento, Ts = 5 ms, t > 20 s. "adapt" = Shtessel con μ = 0.01 (k1 llega a 200).

| Actitud | k1 = 10 | k1 = 40 | k1 = 120 | adaptativa |
|---|---|---|---|---|
| PD (línea base) | 1.11° · 4.96 cm | | | |
| STA explícito | 0.71° · 1.25 cm | 1.95° · 6.1 cm | 17.8° · se cae | 20.3° · se cae |
| STA implícito | 0.71° · 1.34 cm | 1.38° · 1.7 cm | 3.17° · 1.6 cm (rotores saturados) | 22.8° · se cae |
| STA compensado, N = 8 | 0.68° · 1.73 cm | 0.66° · 1.68 cm | 0.66° · 1.68 cm | 0.66° · 1.68 cm |

(error de actitud RMS · error de posición RMS). El compensado no satura los rotores y su
chattering es de 1 a 2 rad/s por muestra (explícito/implícito: 6 a 170). Con N = 4 el lazo
queda al borde (rotores saturados 40–80 % del tiempo): el horizonte debe cubrir el retardo
puro más el filtro de velocidad, además de τ_m. Con k1 = 10, el explícito da menos error de
posición que el compensado (1.25 frente a 1.73 cm). Figura: `~/corridas/campana_actitud/sensibilidad_ganancia.png`.

## Cámara del dron

Con `mision:=busqueda` la ventana trae solo el panel "Detección YOLO" (`config/gazebo_gui_yolo.config`); en las demás misiones trae el panel "Cámara del dron" (a la derecha), que muestra en vivo la
cámara frontal del dron (640×480, 15 fps, inclinada 30° hacia abajo). Lo define
`config/gazebo_gui.config` (la configuración por omisión de Gazebo Harmonic más ese panel),
que el launch pasa con `--gui-config`. El botón ▢ del panel lo separa en una ventana
flotante que se puede agrandar. Fuera de Gazebo: `ros2 run rqt_image_view rqt_image_view
/cuadrirotor/camara`.

## Tópicos ROS 2

| Tópico | Tipo | Sentido |
|---|---|---|
| `/cuadrirotor/odometria` | nav_msgs/Odometry (velocidades en marco del cuerpo) | Gazebo → ROS |
| `/cuadrirotor/imu` | sensor_msgs/Imu | Gazebo → ROS |
| `/cuadrirotor/camara` | sensor_msgs/Image 640×480, 15 Hz | Gazebo → ROS |
| `/cuadrirotor/lidar/points` | sensor_msgs/PointCloud2 (LiDAR 3D: 360° × 120 haces, 16 canales de −35° a +15°, 30 m, 10 Hz; marco del sensor) | Gazebo → ROS |
| `/cuadrirotor/motores` | actuator_msgs/Actuators (velocidad de rotor, rad/s) | ROS → Gazebo |
| `/cuadrirotor/viento` | geometry_msgs/Vector3 (fuerza inercial, N) | ROS → Gazebo |
| `/cuadrirotor/referencia`, `/cuadrirotor/error` | PoseStamped, Vector3Stamped | controlador |
| `/cuadrirotor/busqueda/estado` | std_msgs/String | planificador de la búsqueda |
| `/cuadrirotor/busqueda/cobertura` | std_msgs/Float32 (% del bosque visto) | planificador de la búsqueda |
| `/cuadrirotor/busqueda/mapa_cobertura` | nav_msgs/OccupancyGrid (0 = visto, −1 = sin ver; celdas de 10 m) | planificador de la búsqueda |
| `/cuadrirotor/busqueda/ruta` | nav_msgs/Path (franjas del barrido) | planificador de la búsqueda |
| `/cuadrirotor/evasion/obstaculos` | sensor_msgs/PointCloud2 (obstáculos que recuerda la evasión, marco `world`) | planificador de la búsqueda |
| `/cuadrirotor/incendio/gps` | sensor_msgs/NavSatFix (foco confirmado; QoS transient local) | planificador de la búsqueda |
| `/cuadrirotor/incendio/alerta` | std_msgs/String (aviso con GPS y liga a Google Maps; transient local) | planificador de la búsqueda |

Visualizar: `ros2 run plotjuggler plotjuggler`, `ros2 run rqt_image_view rqt_image_view`.

## Mundos

- `incendio`: terreno plano, tres árboles simples y el foco en (8, 5).
- `bosque`: como el del video de ArduPilot (`iris_forest`): 50 robles y 35 pinos de
  OpenRobotics (Gazebo Fuel, licencia CC0, copiados en `models/roble` y `models/pino` para
  funcionar sin internet) y el foco en (8, 5). Se genera con
  `python3 scripts/generar_bosque.py worlds/bosque.sdf` (semilla fija); deja libre el
  despegue, el corredor al foco y el anillo de la órbita, así que la misión `orbita`
  funciona a 3 m entre los árboles. Los robles miden 6.5 m.

- `bosque_grande`: valle boscoso de 420 × 420 m rodeado de montañas, sobre un terreno de
  1.2 × 1.2 km (malla de 4 m, textura propia de 4096²): lomas suaves, tres cerros de 18 a 42 m
  dentro del bosque y una cordillera alrededor de hasta ~215 m, con laderas de roca y nieve en
  las cumbres; cielo con nubes (`<sky>`). 1700 pinos y robles de tamaños distintos con claros de
  pastizal, 140 rocas y peñascos, y 21 animales (venados, lobos y zorros de OpenGameArt, CC0;
  `scripts/preparar_fauna.py`) que pastan y caminan movidos por el nodo `fauna`. La base de
  despegue está al suroeste (−232, −232), fuera del bosque, y el foco va por omisión en una
  ladera al noreste (105, 135); con `foco_x`/`foco_y` se pone en cualquier otro punto. El launch
  arma el mundo al arrancar (`cuadrirotor_sim/bosque_grande.py`) con el foco sobre el terreno y
  un claro quemado de 9 m alrededor, y le agrega un penacho de humo de ~30 m
  (`models/columna_humo`) para que se vea sobre las copas desde lejos. El terreno, los árboles,
  las rocas y la fauna los genera una sola vez (semilla fija)
  `~/.venvs/cad/bin/python scripts/generar_bosque_grande.py`, que deja en
  `models/terreno_bosque_grande` la malla, la textura, la lista de objetos y el modelo digital
  de elevación (`altura.npz`) que usan el planificador, la fauna y el launch
  (`cuadrirotor_sim/terreno.py`). Corre a tiempo real (RTF ≈ 1.0 con la RTX 5060 Ti).

El foco de incendio (`models/foco_incendio`, en todos los mundos) usa el sistema de partículas de
Gazebo: 7 emisores de llama (blanco amarillento → naranja → rojo → transparente), una columna de
humo que crece y se aclara al subir, brasas, troncos carbonizados y una luz naranja. Las texturas
las genera `python3 scripts/generar_fuego.py`. Es solo visual (no colisiona ni afecta el vuelo).
Gazebo solo encuentra la imagen de colores de las partículas con ruta absoluta, por eso el launch
copia el modelo a `/tmp/cuadrirotor_sim_<uid>/modelos` con las rutas resueltas al arrancar.

## Búsqueda del incendio con YOLO26 (`mision:=busqueda`)

    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque controlador:=sta mision:=busqueda

El dron sale lejos del foco (−40, −35), patrulla el bosque a 15 m, lo encuentra con la
cámara y YOLO26, se dirige a él y lo orbita. Es la cadena completa:
imagen → detección → estimación → referencia → control. El planificador nunca usa la
posición verdadera del foco.

| Nodo | Qué hace |
|---|---|
| `detector_yolo` (Python de `~/.venvs/yolo`) | YOLO26n sobre `/cuadrirotor/camara` a ~10 Hz en la GPU; publica `vision_msgs/Detection2DArray` en `/cuadrirotor/detecciones` y la imagen anotada en `/cuadrirotor/camara_yolo` (panel "Detección YOLO" de Gazebo) |
| `busqueda_incendio` | Estados despegue → patrulla → hacia_humo → aproximacion → (verificacion) → descenso → orbita; publica la consigna en `/cuadrirotor/consigna` (`trajectory_msgs/MultiDOFJointTrajectoryPoint`) y el foco estimado en `/cuadrirotor/foco_estimado` |
| `controlador_pd` / `controlador_sta` | Con `mision:=externa` siguen esa consigna |

Estimación: el humo sube por encima de las copas y se ve de lejos; cada detección da un
rumbo hacia el foco. Las llamas solo se ven cerca (los robles alrededor del claro las
tapan); su rayo de cámara se corta con la superficie 1.1 m sobre el suelo y la mediana de esos
cortes es el foco estimado (si no se ven las llamas, se triangulan los rumbos del humo). La
cámara del dron va inclinada 30° hacia abajo para ver hasta el horizonte durante la patrulla.
Si al llegar al anillo de órbita la estimación sigue saliendo solo del humo, el dron rodea el
punto a 35 m y a la altura de búsqueda (`verificacion`) hasta ver las llamas; solo con llamas
baja a orbitar. Si no las ve en 2 min, descarta la estimación y sigue la patrulla. (Antes
bastaba el humo: en una corrida con la consigna corrida, una triangulación mal condicionada
confirmó un foco a 70 m del verdadero.)

**Aviso con las coordenadas GPS.** Al empezar la órbita (incendio confirmado con las llamas
y visto de cerca) el planificador publica una sola vez la posición del foco en
`/cuadrirotor/incendio/gps` (`NavSatFix`, con la dispersión de los cortes de las llamas como
incertidumbre) y el aviso de texto en `/cuadrirotor/incendio/alerta`; lo escribe en la
terminal y agrega una línea a `~/incendios_detectados.csv` (parámetro `registro_incendios`):

    INCENDIO CONFIRMADO 2026-10-23 04:26:47
      GPS: 19.294219, -99.310000  (19°17'39.2"N 99°18'36.0"W)
      Altitud del suelo: 3019 m   incertidumbre: ±1.0 m
      Mapa: https://www.google.com/maps?q=19.294219,-99.310000
      Local: (105.1, 135.0) m

Para recibirlo en otra terminal: `ros2 topic echo /cuadrirotor/incendio/alerta`. El origen del
mundo está georreferenciado en `cuadrirotor_sim/geo.py` (`ORIGEN`, por omisión el Parque Nacional
Desierto de los Leones, 19.2930° N, 99.3110° O, 3000 m; un lugar supuesto, el relieve no es
el real; parámetro `origen_gps` para cambiarlo). La conversión ENU → WGS84 es exacta
(elipsoide WGS84 vía ECEF), `bosque_grande` declara las mismas `<spherical_coordinates>` para
Gazebo y el *home* del SITL de ArduPilot es el punto de salida en esas coordenadas, así que el
mapa de MAVProxy muestra el dron en el mismo lugar.

**Evasión de obstáculos** (`cuadrirotor_sim/evasion.py`). Toda consigna salvo la órbita pasa
por dos filtros:

- *Relieve* (DEM): con `altura_seguridad` > 0 (10 m en `bosque_grande`) el dron no avanza
  mientras esté a menos de esa altura sobre el terreno que tiene 25 m adelante; primero sube.
- *Árboles, rocas, animales* (LiDAR): los puntos del LiDAR pasan al marco del mundo, se quitan
  los que están a menos de 1.5 m del DEM (son el suelo) y se guardan en vóxeles de 1 m hasta
  60 m del dron. Si el tramo recto hacia el objetivo pasa a menos de `radio_evasion` (3 m) de
  un obstáculo a la altura del dron (±3 m), un histograma polar de 72 sectores (como VFH)
  marca los sectores bloqueados por los puntos a menos de `alcance_evasion` (20 m),
  ensanchados por asin(radio/d), y el dron toma el sector libre más cercano al rumbo del
  objetivo; si todos están bloqueados, sube. Al bajar no entra a menos de 3 m de las copas que
  tiene debajo, y en la órbita sube si el LiDAR ve algo más alto bajo el anillo. Con
  `evasion:=false` (parámetro) se desactiva.

Prueba a 5 m sobre el suelo entre árboles de hasta 8.2 m (`altura_busqueda: 5`, sin GUI):
540 m de patrulla con 12 desvíos, ninguna muestra dentro de una copa, holgura mínima 0.5 m
a la copa más cercana, a unos 2.2 m/s. A los 30 m de la búsqueda normal los árboles no
estorban y la evasión no actúa; con el LiDAR el tiempo real baja de ~1.0 a ~0.8.

### En el bosque grande (`mundo:=bosque_grande`)

    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=sta mision:=busqueda

Parámetros en `config/mundo_bosque_grande.yaml` (el launch lo agrega al planificador, junto con
el DEM del terreno):

- **Barrido en serpentina** (`patron: serpentina`). El planificador cubre la zona del bosque
  (−210 a 210 m en x e y) avanzando hacia el frente del dron (su rumbo al despegar; en la base
  mira a +x) con barridos de lado a lado separados 70 m, unidos por semicírculos de 35 m de
  radio (2.6 km, ~9 min a 5 m/s). El dron persigue un punto 10 m adelante sobre la ruta
  (`anticipo_ruta`), así que no frena en las vueltas. Si termina sin ver el incendio, recorre la
  serpentina de regreso y sigue vigilando. Con `patron: franjas` vuelve al barrido anterior
  (franjas este-oeste con vueltas en ángulo recto).
- **Seguimiento del relieve.** Vuela a 30 m sobre el suelo: toma la altura máxima del terreno
  entre la consigna y 50 m hacia adelante, así que empieza a subir antes de llegar a un cerro
  (hasta 3 m/s en vertical). El planificador conoce el relieve (el DEM, como un dron real con
  un mapa de elevación), pero no dónde está el incendio.
- **Estimación.** El rayo de la cámara hacia las llamas se corta con el relieve, no con un
  plano.
- **Órbita.** Por encima de las copas: 18 m sobre el suelo del foco y al menos 14 m sobre el
  punto más alto del anillo, con radio de 14 m (el planificador no sabe dónde hay árboles).
- **Cobertura.** Cada 0.5 s se proyecta una rejilla de 24 × 18 rayos de la cámara sobre el
  relieve (hasta 150 m) y se marcan las celdas de 10 m que ya se vieron. Se publica el
  porcentaje en `/cuadrirotor/busqueda/cobertura`, el mapa en `/cuadrirotor/busqueda/mapa_cobertura` y
  la ruta en `/cuadrirotor/busqueda/ruta` (se ven en RViz con el marco `world`).
- **Ventana.** Arranca mirando la base y a los 12 s la cámara sigue al dron.

Corridas de verificación (23 oct 2026, `tarot650`, STA implícito, sin viento, YOLO reentrenado):

| Foco | Primera detección | Estimación final (error) | Patrulla → órbita | Cobertura al encontrarlo |
|---|---|---|---|---|
| (105, 135), por omisión | humo, triangulación (103.5, 134.2) | (105.1, 135.0) (0.1 m) | 2 min 42 s | 60% |
| (−150, 60) | llamas (−149.4, 62.4) | (−150.0, 59.9) (0.1 m) | 1 min 8 s | 29% |

Altura sobre el suelo durante el vuelo: mediana 30.3 m, mínima 13.9 m (sobre la ladera del
anillo de órbita); error de seguimiento del STA 5 cm RMS (máx. 22 cm). En el mundo `bosque`
la búsqueda sigue igual (foco estimado en (8.3, 5.2)). Con las llamas solo se usan cortes a
menos de 100 m de la cámara (`alcance_llamas`): más lejos el rayo roza el suelo y la primera
estimación se iba hasta 50 m.

**Modelo YOLO.** YOLO26 es la última versión publicada de Ultralytics (YOLO27 está
anunciado pero sin pesos). Los modelos preentrenados en COCO no tienen clase de fuego ni de
humo, así que se ajustó YOLO26n con imágenes del simulador etiquetadas automáticamente:

    # 1) datasets (Gazebo con el mundo de captura, ver el encabezado del script)
    python3 scripts/capturar_dataset.py capturar ~/datasets/incendio_bosque 3000
    python3 scripts/capturar_dataset.py capturar ~/datasets/incendio_bosque_grande 3000 bosque_grande
    # 2) entrenamiento en la GPU con los dos; copia el mejor modelo a yolo/incendio_yolo26n.pt
    ~/.venvs/yolo/bin/python scripts/entrenar_yolo.py ~/datasets/incendio_bosque ~/datasets/incendio_bosque_grande 60

`capturar_dataset.py` mueve una cámara con la óptica del dron y el foco por el bosque,
y etiqueta cada imagen proyectando el volumen de las llamas y de la columna de humo; los
puntos que tapan las copas (y, en `bosque_grande`, el relieve) se descartan. En `bosque_grande`
una de cada cinco tomas apunta a una roca o a un animal: con el modelo entrenado solo en el
bosque plano, una roca oscura de 2.6 m se detectaba como fuego (se parece al suelo quemado
del foco). Entorno de YOLO (una vez; `instalar_vision_yolo.sh`
viene junto a los otros scripts de instalación):

    bash instalar_vision_yolo.sh

Crea `~/.venvs/yolo` con `--system-site-packages` (para ver ROS 2), instala `ultralytics`
con `numpy<2` (los mensajes de ROS 2 Jazzy están compilados con numpy 1.x),
`opencv-python<4.12` y `setuptools<80`, y comprueba la GPU y el modelo.

Para volar con dron real conviene volver a entrenar mezclando imágenes reales de incendios
(por ejemplo, el dataset D-Fire) con las del simulador.

## Con ArduPilot SITL (`controlador:=ardupilot`)

    ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque controlador:=ardupilot mision:=orbita viento:=true

El dron lo pilota ArduCopter 4.7.2 SITL en lugar de los controladores de ROS 2:

- `models/tarot650_ardupilot` (generado por `generar_modelo.py`) es el mismo Tarot con el
  plugin `ArduPilotPlugin`: cada salida de motor (1000–2000 µs) se publica como velocidad
  de rotor (0–680 rad/s) y el sistema `ComandosMotores` (paquete `cuadrirotor_gz_plugins`)
  las junta para `MulticopterMotorModel`. La física de motores, el viento y el incendio son
  los mismos que con el PD y el STA.
- Parámetros de ArduCopter: `config/ardupilot/tarot650.parm` (quad X, curva de empuje
  cuadrática `MOT_THST_EXPO 1`, `MOT_THST_HOVER 0.28`, lazos de tasa más suaves que el Iris).
- `mision_ardupilot` espera al EKF, arma en GUIDED, despega a la altura de la misión y manda
  la misma referencia (`dinamica.Referencia`) por MAVLink. Guarda el CSV con `csv:=`, en el
  formato de `metricas.py`. Corre con el Python de `~/venv-ardupilot` (pymavlink).
- El viento se sincroniza con el inicio de la misión (antes hay ~30 s de espera del EKF).
- Con `gui:=true` se abren las ventanas del SITL: la terminal de ArduCopter, una terminal
  `MAVProxy` con su prompt (ahí se le pueden escribir comandos: `mode RTL`, `arm`, ...) y la
  consola y el mapa de MAVProxy. Reenvía a UDP 14550 para QGroundControl.
- Con `mision:=busqueda`, `mision_ardupilot` despega a 5 m y después reenvía a ArduPilot las
  consignas de `busqueda_incendio`. Con el backend JSON el marco local de ArduPilot es el del
  mundo de Gazebo (N = y, E = x) y su GPS es el *home* más la posición en Gazebo, así que el
  *home* del SITL es el origen del mundo (`geo.ORIGEN`) y el GPS de ArduPilot coincide con el del
  aviso. Antes de armar se mide el desfase entre LOCAL_POSITION_NED y la odometría (sale ~0) y
  se descuenta. (Una primera versión restaba el punto de salida: el dron volaba 330 m corrido y
  llegó a estrellarse contra un cerro.)
- `ventanas_sitl.py` acomoda las ventanas en columnas cuando aparecen (terminal MAVProxy,
  consola y terminal ArduCopter a la izquierda; Gazebo al centro; mapa a la derecha) y deja al
  frente la terminal de MAVProxy. Usa `python-xlib` (instalado en `~/venv-ardupilot`).
- El módulo de MAVProxy `mavproxy_trayectoria.py` (se carga solo con GUI) dibuja en el mapa la
  trayectoria recorrida (magenta, toda, un punto cada 2 m), la ruta planeada (cian) y, al
  confirmarse, el foco del incendio (círculo rojo con sus coordenadas). La ruta y el foco los
  escribe `busqueda_incendio` en `/tmp/cuadrirotor_sim_<uid>/mapa_busqueda.json`. En el prompt
  de MAVProxy: `trayectoria borrar`, `trayectoria ocultar`, `trayectoria mostrar`.
- Al cerrar el launch (Ctrl-C) un manejador termina lo que no se cierra solo: el `gz sim` detrás
  del envoltorio de ruby y MAVProxy/ArduCopter.

Búsqueda con ArduPilot (23 oct 2026, `bosque_grande`, foco por omisión, serpentina): humo
triangulado en (111.2, 133.6), llamas y foco final en (105.1, 135.0), órbita a 22 m sobre el
suelo con 79 % del bosque visto.
- Rutas: `ARDUPILOT_DIR` (por omisión `~/ardupilot`) y `ARDUPILOT_VENV` (`~/venv-ardupilot`).

Resultado (bosque, órbita con viento 1 ± 0.5 N): error RMS 13.7 cm con ArduPilot, frente a
11.9 cm del PD y 4.7 cm del STA adaptativo.

## Vehículos

Los dos se generan con `python3 scripts/generar_modelo.py`, que escribe `models/<vehiculo>/model.sdf`
y `config/vehiculo_<vehiculo>.yaml` desde la misma tabla, así que el controlador siempre usa los
mismos parámetros que la planta. En Gazebo ambos se llaman `cuadrirotor` y usan los mismos tópicos.

| | `cuadrirotor` | `tarot650` |
|---|---|---|
| Configuración | "+" | "X" (motores en ±0.2198, ±0.212 m; frente paralelo a los patines) |
| Masa | 1.5 kg | 2.2 kg (supuesta) |
| Inercia total | 0.0295, 0.0295, 0.056 kg·m² | 0.0361, 0.0341, 0.0660 kg·m² (estimada) |
| kf · ω_max | 8.55e-6 · 1000 rad/s | 4.24e-5 · 680 rad/s (supuestos; ≈ 20 N por motor) |
| Retardo de motores | 12.5 / 25 ms | 30 / 50 ms (supuesto) |
| Vuelo estacionario | 656 rad/s | 357 rad/s |
| STA: μ (x, y, z) · k1_max | 0.12, 0.12, 0.02 · 8 | 0.3, 0.3, 0.03 · 3 |

**Tarot Ironman 650.** Mallas convertidas del CAD de GrabCAD
(https://grabcad.com/library/tarot-ironman-650-2, motores T-Motor MN3508, hélices 15×5): marco,
campanas de los motores y las 4 hélices. La geometría (posición de motores, tren de aterrizaje)
es la del CAD; masa, inercias, constantes de hélice y retardos son valores típicos supuestos y
conviene reemplazarlos por los del dron real (peso con batería, KV, hélice, celdas). Para regenerar
las mallas:

    ~/.venvs/cad/bin/python scripts/cad_a_malla.py exportar "Tarot 650/Tarot.stp" salida/ \
        --grupo placas='Base$|base|side bracket|lower support|bracket landing' \
        --grupo brazos='CarbonArm' --grupo acentos='orange|clamp|pivot|clip|Motortoarm|ArmCopter|cap' \
        --grupo tren='Lower Tube|leg|foot|landing-gear|Landing' --grupo motores='bell' \
        --grupo marco=resto --excluir '94180|bearing|circlip|coil|axle|stator|15x5' \
        --ejes +z +x +y --origen 0.0142 0 0.0070

No simplifiques `placas` ni `brazos`: las placas de carbono miden 1.6 mm y la simplificación
las deshace. Sí se pueden reducir `acentos` (a 50 000 caras), `tren` (25 000) y `motores` (20 000).
Las hélices se exportan aparte con `--grupo helice='15x5' --separar helice --caras 4000`; `helice_k` se
asigna al rotor por posición.

Con viento, en el bosque, la órbita da 11.9 cm de error RMS con el PD y 4.7 cm con el STA adaptativo
implícito. El retardo mayor de sus motores agranda el ciclo límite del STA en x, y (|s| ≈ 0.2–0.35
m/s); con el μ del cuadrirrotor la ganancia crece hasta inestabilizarlo, por eso μ es por vehículo.

## Modelo

Configuración "+", m = 1.5 kg, l = 0.225 m, J ≈ diag(0.0295, 0.0295, 0.056) kg·m²,
kf = 8.54858e-6 N/(rad/s)², km = 0.016 m, ω_max = 1000 rad/s, motores de primer orden
(τ_subida 12.5 ms, τ_bajada 25 ms). Rotores 0 (+x, ccw), 1 (+y, cw), 2 (−x, ccw), 3 (−y, cw).

Para cambiar parámetros edita la tabla `VEHICULOS` de `scripts/generar_modelo.py` y vuelve a
ejecutarlo; regenera el modelo y su YAML.

## Agregar un controlador nuevo

Hereda de `ControladorPD` y sobrescribe `aceleracion(ep, ev, ad)` (ver `controlador_sta.py`);
regístralo en `setup.py` (`entry_points`). `dinamica.py` contiene el mezclador, las rotaciones
y la referencia de la misión.

## Pruebas

    cd src/cuadrirotor_sim && python3 -m pytest -q test/

## Paquetes

- `cuadrirotor_sim`: modelo, mundo, puente, controladores PD y STA, viento, launch, scripts.
- `cuadrirotor_gz_plugins`: sistema de Gazebo `FuerzaViento` (fuerza externa retenida
  en cada paso de física).
