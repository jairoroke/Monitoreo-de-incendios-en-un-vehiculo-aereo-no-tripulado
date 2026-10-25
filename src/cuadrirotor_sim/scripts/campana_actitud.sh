#!/usr/bin/env bash
# Campaña del STA en el lazo de actitud del Tarot 650 (órbita con viento).
# Uso: scripts/campana_actitud.sh <directorio> [duración_s]
# Esquemas de actitud x k1 fijo, y ganancia adaptativa con mu pequeño.
DIR=$1; DUR=${2:-45}; mkdir -p $DIR
export VEHICULO=${VEHICULO:-tarot650}
C=$(dirname "$0")/corrida.sh
for k in 10.0 40.0 120.0; do
  $C $DIR/exp_k${k%.0}.csv $DUR actitud=sta esquema_actitud=explicito k1_actitud=$k
  $C $DIR/imp_k${k%.0}.csv $DUR actitud=sta esquema_actitud=implicito k1_actitud=$k
  $C $DIR/compN4_k${k%.0}.csv $DUR actitud=sta esquema_actitud=compensado horizonte_actitud=4 k1_actitud=$k
  $C $DIR/compN8_k${k%.0}.csv $DUR actitud=sta esquema_actitud=compensado horizonte_actitud=8 k1_actitud=$k
done
for e in explicito implicito; do
  $C $DIR/${e:0:3}_adapt.csv $DUR actitud=sta esquema_actitud=$e adaptativo_actitud=true mu_actitud=[0.01,0.01,0.01] k1_actitud=10.0
done
$C $DIR/compN4_adapt.csv $DUR actitud=sta esquema_actitud=compensado horizonte_actitud=4 adaptativo_actitud=true mu_actitud=[0.01,0.01,0.01] k1_actitud=10.0
$C $DIR/compN8_adapt.csv $DUR actitud=sta esquema_actitud=compensado horizonte_actitud=8 adaptativo_actitud=true mu_actitud=[0.01,0.01,0.01] k1_actitud=10.0
$C $DIR/pd.csv $DUR
