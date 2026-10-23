"""Lanza Gazebo con el escenario del incendio, el puente ROS 2 y el controlador.

Ejemplos:
  ros2 launch cuadrirotor_sim simulacion.launch.py
  ros2 launch cuadrirotor_sim simulacion.launch.py mision:=orbita viento:=true
  ros2 launch cuadrirotor_sim simulacion.launch.py gui:=false csv:=/tmp/corrida.csv
  ros2 launch cuadrirotor_sim simulacion.launch.py controlador:=sta esquema:=implicito
  ros2 launch cuadrirotor_sim simulacion.launch.py mundo:=bosque mision:=orbita
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque controlador:=ardupilot mision:=orbita
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque controlador:=sta mision:=busqueda
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=sta mision:=busqueda
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=sta mision:=busqueda foco_x:=-150 foco_y:=60
  ros2 launch cuadrirotor_sim simulacion.launch.py vehiculo:=tarot650 mundo:=bosque_grande controlador:=ardupilot mision:=busqueda

En 'bosque_grande' el mundo se arma al lanzar (cuadrirotor_sim/bosque_grande.py)
con el foco en (foco_x, foco_y) sobre el terreno, el dron sale de la base al
suroeste del valle y el nodo 'fauna' mueve a los animales.

Con controlador:=ardupilot el dron lo pilota ArduCopter SITL (~/ardupilot, o la
ruta de ARDUPILOT_DIR) con el Python de ~/venv-ardupilot (o ARDUPILOT_VENV), y
el nodo mision_ardupilot le manda la misma referencia que a los controladores
de ROS 2 (con mision:=busqueda, las consignas de busqueda_incendio). Con GUI se
abren las ventanas del SITL: la terminal de ArduCopter y la consola y el mapa
de MAVProxy, acomodadas por ventanas_sitl.py. El 'home' del SITL es el origen
del mundo (geo.ORIGEN), así que su GPS coincide con el del aviso del incendio.
"""
import os
import signal
import tempfile
import time

import yaml

from ament_index_python.packages import (PackageNotFoundError, get_package_prefix,
                                          get_package_share_directory)
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument, ExecuteProcess,
                            RegisterEventHandler,
                            IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from cuadrirotor_sim import bosque_grande
from cuadrirotor_sim.geo import ORIGEN
from cuadrirotor_sim.terreno import Terreno


def dir_temporal(*partes):
    d = os.path.join(tempfile.gettempdir(), f'cuadrirotor_sim_{os.getuid()}', *partes)
    os.makedirs(d, exist_ok=True)
    return d


def terminar_restos(event, context):
    """Al cerrar, termina lo que el launch no alcanza a cerrar solo.

    'gz sim' corre detrás de un envoltorio de ruby que no reenvía la señal, y
    sim_vehicle (dentro de su terminal) no cierra MAVProxy ni ArduCopter cuando
    se cierra la terminal. Solo toca los procesos de esta simulación: el gz sim
    con un mundo de cuadrirotor_sim (del paquete o de /tmp; y sus hijos), el SITL de
    ArduCopter, el MAVProxy conectado a él y la terminal 'ArduCopter'.
    """
    procesos = {}
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='replace')
            with open(f'/proc/{pid}/stat') as f:
                ppid = int(f.read().rsplit(')', 1)[1].split()[1])
            if os.stat(f'/proc/{pid}').st_uid == os.getuid():
                procesos[int(pid)] = (ppid, cmd)
        except (OSError, ValueError, IndexError):
            continue
    raices = {p for p, (_, c) in procesos.items()
              if (c.startswith('gz sim') and 'cuadrirotor_sim' in c)
              or 'build/sitl/bin/arducopter' in c
              or ('mavproxy.py' in c and 'tcp:127.0.0.1:5760' in c)
              or (c.startswith('xterm') and '-T ArduCopter' in c)}
    todos, nuevos = set(raices), set(raices)
    while nuevos:
        nuevos = {p for p, (pp, _) in procesos.items() if pp in nuevos} - todos
        todos |= nuevos
    todos.discard(os.getpid())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        vivos = []
        for p in todos:
            try:
                os.kill(p, sig)
                vivos.append(p)
            except OSError:
                pass
        if not vivos:
            break
        time.sleep(2.0)


def ruta_dem(share, mundo):
    """DEM del mundo, o '' si el suelo es el plano z = 0."""
    ruta = os.path.join(share, 'models', f'terreno_{mundo}', 'altura.npz')
    return ruta if os.path.exists(ruta) else ''


def foco_pedido(context):
    fx, fy = (LaunchConfiguration(k).perform(context) for k in ('foco_x', 'foco_y'))
    if bool(fx) != bool(fy):
        raise RuntimeError('Da foco_x y foco_y juntos')
    return (float(fx), float(fy)) if fx else None


def punto_salida(context, share):
    """(x, y, altura del suelo) donde aparece el vehículo."""
    # En la búsqueda el dron sale lejos del foco, en un claro al borde del bosque;
    # en 'bosque_grande', de la base (una losa de 5 cm sobre el terreno).
    mundo = LaunchConfiguration('mundo').perform(context)
    x0, y0 = LaunchConfiguration('x0').perform(context), LaunchConfiguration('y0').perform(context)
    if mundo == 'bosque_grande':
        bx, by = bosque_grande.datos(os.path.join(share, 'models'))[0]['base']
        x0, y0 = x0 or str(bx), y0 or str(by)
    elif LaunchConfiguration('mision').perform(context) == 'busqueda':
        x0, y0 = x0 or '-40', y0 or '-35'
    x0, y0 = float(x0 or 0), float(y0 or 0)
    dem = ruta_dem(share, mundo)
    return x0, y0, Terreno(dem).altura(x0, y0) + 0.05 if dem else 0.0


def aparecer_vehiculo(context, share):
    """Agrega el vehículo elegido al mundo, siempre con el nombre 'cuadrirotor'."""
    vehiculo = LaunchConfiguration('vehiculo').perform(context)
    with open(os.path.join(share, 'config', f'vehiculo_{vehiculo}.yaml')) as f:
        z0 = yaml.safe_load(f)['/**']['ros__parameters']['z_inicial']
    modelo = vehiculo
    if LaunchConfiguration('controlador').perform(context) == 'ardupilot':
        modelo = f'{vehiculo}_ardupilot'
        if not os.path.isdir(os.path.join(share, 'models', modelo)):
            raise RuntimeError(f"El vehículo '{vehiculo}' no tiene variante para ArduPilot")
    x0, y0, suelo = punto_salida(context, share)
    z0 += suelo
    return [Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-world', LaunchConfiguration('mundo').perform(context),
                   '-file', os.path.join(share, 'models', modelo, 'model.sdf'),
                   '-name', 'cuadrirotor', '-x', str(x0), '-y', str(y0), '-z', f'{z0:.3f}'])]


def busqueda(context, share):
    """Detector YOLO26 y planificador de la búsqueda del incendio."""
    if LaunchConfiguration('mision').perform(context) != 'busqueda':
        return []
    vehiculo = LaunchConfiguration('vehiculo').perform(context)
    venv = os.path.expanduser(os.environ.get('YOLO_VENV', '~/.venvs/yolo'))
    python = os.path.join(venv, 'bin', 'python3')
    modelo = LaunchConfiguration('modelo_yolo').perform(context) or \
        os.path.join(share, 'yolo', 'incendio_yolo26n.pt')
    for ruta, que in ((python, 'entorno de Python de YOLO'), (modelo, 'modelo YOLO')):
        if not os.path.exists(ruta):
            raise RuntimeError(f'No se encontró el {que}: {ruta}')
    detector = ExecuteProcess(
        cmd=[python, '-m', 'cuadrirotor_sim.detector_yolo', '--ros-args',
             '-p', 'use_sim_time:=true', '-p', f'modelo:={modelo}'],
        output='screen', name='detector_yolo')
    mundo = LaunchConfiguration('mundo').perform(context)
    config_mundo = os.path.join(share, 'config', f'mundo_{mundo}.yaml')
    # Ruta y foco para el mapa de MAVProxy (mavproxy_trayectoria); sin restos de otra corrida.
    archivo_mapa = os.path.join(dir_temporal(), 'mapa_busqueda.json')
    if os.path.exists(archivo_mapa):
        os.remove(archivo_mapa)
    planificador = Node(
        package='cuadrirotor_sim', executable='busqueda_incendio', output='screen',
        parameters=[os.path.join(share, 'config', 'parametros.yaml'),
                    os.path.join(share, 'config', f'vehiculo_{vehiculo}.yaml'),
                    *([config_mundo] if os.path.exists(config_mundo) else []),
                    {'use_sim_time': True, 'terreno': ruta_dem(share, mundo),
                     'archivo_mapa': archivo_mapa}])
    return [detector, planificador]


def ardupilot(context, share):
    """ArduCopter SITL (sim_vehicle.py con MAVProxy) y el nodo de misión."""
    if LaunchConfiguration('controlador').perform(context) != 'ardupilot':
        return []
    vehiculo = LaunchConfiguration('vehiculo').perform(context)
    ap = os.path.expanduser(os.environ.get('ARDUPILOT_DIR', '~/ardupilot'))
    venv = os.path.expanduser(os.environ.get('ARDUPILOT_VENV', '~/venv-ardupilot'))
    python = os.path.join(venv, 'bin', 'python3')
    for ruta, que in ((os.path.join(ap, 'build', 'sitl', 'bin', 'arducopter'), 'ArduCopter SITL compilado'),
                      (python, 'entorno de Python de ArduPilot')):
        if not os.path.exists(ruta):
            raise RuntimeError(f'No se encontró el {que}: {ruta}')
    trabajo = os.path.join(tempfile.gettempdir(), f'cuadrirotor_sim_{os.getuid()}', 'ardupilot')
    os.makedirs(trabajo, exist_ok=True)

    sitl = [python, os.path.join(ap, 'Tools', 'autotest', 'sim_vehicle.py'),
            '-v', 'ArduCopter', '-f', 'gazebo-iris', '--model', 'JSON', '--no-rebuild', '-w',
            '--add-param-file', os.path.join(share, 'config', 'ardupilot', f'{vehiculo}.parm'),
            '--out', 'udp:127.0.0.1:14551']
    # El SITL (backend JSON) pone el 'home' en el origen del mundo de Gazebo y su GPS es
    # home + posición en Gazebo: así coincide con geo.enu_a_geodesica.
    lat, lon, alt = ORIGEN
    sitl += ['-l', f'{lat:.7f},{lon:.7f},{alt:.1f},0']
    gui = LaunchConfiguration('gui').perform(context) == 'true'
    if gui:   # con la trayectoria recorrida dibujada en el mapa
        sitl += ['--console', '--map',
                 '--mavproxy-args=--load-module=cuadrirotor_sim.mavproxy_trayectoria']
    else:
        sitl += ['--mavproxy-args=--non-interactive']
    entorno = {'PATH': os.path.join(venv, 'bin') + os.pathsep + os.environ.get('PATH', ''),
               'VIRTUAL_ENV': venv}
    # La terminal de ArduCopter, a la vista abajo a la izquierda (sin tapar la de MAVProxy).
    entorno['SITL_RITW_TERMINAL'] = 'xterm -T ArduCopter -geometry 120x18+0-0 -fa Monospace -fs 9 -hold -e'

    mision = [python, '-m', 'cuadrirotor_sim.mision_ardupilot', '--ros-args',
              '--params-file', os.path.join(share, 'config', 'parametros.yaml'),
              '-p', 'use_sim_time:=true',
              '-p', f"mision:={LaunchConfiguration('mision').perform(context)}"]
    if LaunchConfiguration('mision').perform(context) == 'busqueda':
        mision += ['-p', 'altura:=5.0']   # despegue; después manda busqueda_incendio
    csv = LaunchConfiguration('csv').perform(context)
    if csv:
        mision += ['-p', f'archivo_csv:={csv}']
    if gui:   # sim_vehicle en su propia terminal: ahí queda el prompt de MAVProxy
        sitl = ['xterm', '-T', 'MAVProxy: escribe aquí los comandos (arm throttle, mode GUIDED, ...)',
                '-geometry', '110x32+0+0', '-fa', 'Monospace', '-fs', '11', '-e', *sitl]
    ventanas = [ExecuteProcess(cmd=[python, '-m', 'cuadrirotor_sim.ventanas_sitl', '120'],
                               additional_env=entorno, output='screen', name='ventanas_sitl')] if gui else []
    return [*ventanas,
            ExecuteProcess(cmd=sitl, cwd=trabajo, additional_env=entorno, output='screen',
                           name='ardupilot_sitl'),
            ExecuteProcess(cmd=mision, additional_env=entorno, output='screen',
                           name='mision_ardupilot')]


def modelos_con_rutas_absolutas(share):
    """Copias de foco_incendio y columna_humo con rutas absolutas a sus texturas.

    Gazebo solo encuentra la imagen de rango de colores de las partículas
    (<color_range_image>) con una ruta absoluta; la ruta depende de dónde está
    instalado el paquete, así que se resuelve al lanzar.
    """
    texturas = os.path.join(share, 'models', 'foco_incendio')
    destino = dir_temporal('modelos')
    for nombre in ('foco_incendio', 'columna_humo'):
        origen = os.path.join(share, 'models', nombre)
        os.makedirs(os.path.join(destino, nombre), exist_ok=True)
        with open(os.path.join(origen, 'model.sdf')) as f:
            sdf = f.read().replace('model://foco_incendio/', texturas + '/')
        with open(os.path.join(destino, nombre, 'model.sdf'), 'w') as f:
            f.write(sdf)
        with open(os.path.join(origen, 'model.config')) as f, \
                open(os.path.join(destino, nombre, 'model.config'), 'w') as g:
            g.write(f.read())
    return destino


def gazebo(context, share):
    """Gazebo con el mundo elegido; 'bosque_grande' se arma aquí con el foco pedido."""
    mundo = LaunchConfiguration('mundo').perform(context)
    gui = LaunchConfiguration('gui').perform(context) == 'true'
    mision = LaunchConfiguration('mision').perform(context)
    archivo = os.path.join(share, 'worlds', f'{mundo}.sdf')
    ventana = os.path.join(share, 'config', 'gazebo_gui' + ('_yolo' if mision == 'busqueda' else '') + '.config')
    acciones = []
    if mundo == 'bosque_grande':
        modelos = os.path.join(share, 'models')
        archivo = bosque_grande.escribir(modelos, os.path.join(dir_temporal(), 'bosque_grande.sdf'),
                                         foco_pedido(context))
        # La ventana arranca mirando la base y luego sigue al dron.
        bx, by = bosque_grande.datos(modelos)[0]['base']
        with open(ventana) as f:
            cfg = f.read().replace('<camera_pose>-6 0 6 0 0.5 0</camera_pose>',
                                   f'<camera_pose>{bx - 14:g} {by - 14:g} 9 0 0.35 0.785</camera_pose>')
        ventana = os.path.join(dir_temporal(), os.path.basename(ventana))
        with open(ventana, 'w') as f:
            f.write(cfg)
        acciones.append(Node(package='cuadrirotor_sim', executable='fauna', output='screen',
                             arguments=[mundo, modelos]))
        if gui:
            seguir = ['gz', 'service', '--reptype', 'gz.msgs.Boolean', '--timeout', '3000']
            acciones.append(TimerAction(period=12.0, actions=[
                ExecuteProcess(cmd=seguir + ['-s', '/gui/follow/offset', '--reqtype', 'gz.msgs.Vector3d',
                                             '--req', 'x: -12, y: -12, z: 7'], output='log'),
                ExecuteProcess(cmd=seguir + ['-s', '/gui/follow', '--reqtype', 'gz.msgs.StringMsg',
                                             '--req', 'data: "cuadrirotor"'], output='log')]))
    gz_args = f'-r --gui-config {ventana} {archivo}' if gui else f'-r -s --headless-rendering {archivo}'
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args, 'on_exit_shutdown': 'true'}.items()), *acciones]


def ruta_plugin_ardupilot():
    """Agrega el plugin de ardupilot_gazebo a Gazebo si está compilado en el workspace."""
    try:
        prefijo = get_package_prefix('ardupilot_gazebo')
    except PackageNotFoundError:
        return []
    return [AppendEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
                                      os.path.join(prefijo, 'lib', 'ardupilot_gazebo'))]


def generate_launch_description():
    share = get_package_share_directory('cuadrirotor_sim')

    # Con GPU NVIDIA, Mesa intenta abrir el dispositivo por EGL y llena la
    # terminal de avisos 'libEGL warning'; se fuerza el EGL de NVIDIA.
    egl_nvidia = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    entorno_egl = ([SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES', egl_nvidia)]
                   if os.path.exists(egl_nvidia) else [])
    vehiculo_yaml = [os.path.join(share, 'config', 'vehiculo_'), LaunchConfiguration('vehiculo'), '.yaml']

    puente = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': os.path.join(share, 'config', 'puente.yaml'),
                     'use_sim_time': True}])

    controlador = Node(
        package='cuadrirotor_sim', output='screen',
        executable=['controlador_', LaunchConfiguration('controlador')],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('controlador'), "' not in ('ninguno', 'ardupilot')"])),
        parameters=[os.path.join(share, 'config', 'parametros.yaml'), vehiculo_yaml, {
            'use_sim_time': True,
            'mision': PythonExpression(["'externa' if '", LaunchConfiguration('mision'),
                                        "' == 'busqueda' else '", LaunchConfiguration('mision'), "'"]),
            'Ts': LaunchConfiguration('Ts'),
            'archivo_csv': LaunchConfiguration('csv'),
            'esquema': LaunchConfiguration('esquema'),
        }])

    viento = Node(
        package='cuadrirotor_sim', executable='viento', output='screen',
        condition=IfCondition(LaunchConfiguration('viento')),
        parameters=[os.path.join(share, 'config', 'parametros.yaml'), {
            'use_sim_time': True,
            'sincronizar': PythonExpression(["'", LaunchConfiguration('controlador'), "' == 'ardupilot' or '",
                                             LaunchConfiguration('mision'), "' == 'busqueda'"]),
        }])

    return LaunchDescription([
        DeclareLaunchArgument('mundo', default_value='incendio', description='incendio | bosque | bosque_grande'),
        DeclareLaunchArgument('vehiculo', default_value='cuadrirotor', description='cuadrirotor | tarot650'),
        DeclareLaunchArgument('gui', default_value='true', description='Mostrar la interfaz de Gazebo'),
        DeclareLaunchArgument('controlador', default_value='pd', description='pd | sta | ardupilot | ninguno (para lanzarlo aparte con ros2 run)'),
        DeclareLaunchArgument('esquema', default_value='implicito',
                              description='STA: explicito | semi_implicito | implicito'),
        DeclareLaunchArgument('mision', default_value='hover',
                              description='hover | orbita | busqueda (YOLO26 encuentra el incendio)'),
        DeclareLaunchArgument('x0', default_value='', description='Posición inicial x [m]'),
        DeclareLaunchArgument('y0', default_value='', description='Posición inicial y [m]'),
        DeclareLaunchArgument('foco_x', default_value='', description='bosque_grande: x del foco [m]'),
        DeclareLaunchArgument('foco_y', default_value='', description='bosque_grande: y del foco [m]'),
        DeclareLaunchArgument('modelo_yolo', default_value='', description='Pesos de YOLO (.pt)'),
        DeclareLaunchArgument('Ts', default_value='0.005', description='Periodo de muestreo [s]'),
        DeclareLaunchArgument('viento', default_value='false', description='Aplicar perturbación de viento'),
        DeclareLaunchArgument('csv', default_value='', description='Ruta del CSV de registro'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', modelos_con_rutas_absolutas(share)),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(share, 'models')),
        *entorno_egl,
        *ruta_plugin_ardupilot(),
        RegisterEventHandler(OnShutdown(on_shutdown=terminar_restos)),
        OpaqueFunction(function=gazebo, args=[share]),
        OpaqueFunction(function=aparecer_vehiculo, args=[share]), puente, controlador, viento,
        OpaqueFunction(function=ardupilot, args=[share]),
        OpaqueFunction(function=busqueda, args=[share]),
    ])
