#!/usr/bin/env bash
# Una corrida sin interfaz de la misión de órbita con viento y el controlador STA.
# Uso: [VEHICULO=tarot650] [MUNDO=bosque] scripts/corrida.sh <csv> <duración_s> [clave=valor ...]
#   ej. scripts/corrida.sh ~/corridas/imp_ts20.csv 50 esquema=implicito Ts=0.02 adaptativo=false k1_0=4.0
source /opt/ros/jazzy/setup.bash; source "$HOME/ros2_ws/install/setup.bash"
CFGDIR=$(ros2 pkg prefix cuadrirotor_sim)/share/cuadrirotor_sim/config
VEHICULO=${VEHICULO:-cuadrirotor}; MUNDO=${MUNDO:-incendio}
CSV=$1; DUR=$2; shift 2
EXTRA=$(mktemp --suffix=.yaml)
{ echo "controlador_sta:"; echo "  ros__parameters:"
  echo "    use_sim_time: true"; echo "    mision: orbita"; echo "    archivo_csv: $CSV"
  for kv in "$@"; do echo "    ${kv%%=*}: ${kv#*=}"; done; } > $EXTRA
setsid ros2 launch cuadrirotor_sim simulacion.launch.py gui:=false viento:=true controlador:=ninguno vehiculo:=$VEHICULO mundo:=$MUNDO > /dev/null 2>&1 &
L=$!
timeout -s INT $DUR ros2 run cuadrirotor_sim controlador_sta --ros-args --params-file $CFGDIR/parametros.yaml --params-file $CFGDIR/vehiculo_$VEHICULO.yaml --params-file $EXTRA > /dev/null 2>&1
kill -INT -- -$L; sleep 4; kill -KILL -- -$L 2>/dev/null; rm -f $EXTRA; sleep 1
