#!/usr/bin/env python3
"""Genera figuras, tablas y la copia del código del manual de FuegoNet.

    cd ~/ros2_ws/src/cuadrirotor_sim
    ~/.venvs/yolo/bin/python doc/manual_fuegonet/herramientas/generar.py

Lee ~/datasets/fuegonet/ (resultados.csv de las dos corridas, evaluacion.json,
mejor.pt) y escribe en doc/manual_fuegonet/{figuras,capitulos,codigo}. Lo que
todavía no exista se marca como pendiente para que el manual siempre compile.
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MANUAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAQUETE = os.path.dirname(os.path.dirname(MANUAL))
sys.path.insert(0, PAQUETE)
FIG, CAP, COD = (os.path.join(MANUAL, d) for d in ('figuras', 'capitulos', 'codigo'))
RUNS = os.path.expanduser('~/datasets/fuegonet')
DATOS = os.path.expanduser('~/datasets/combinado_incendio_bosque_incendio_bosque_grande/data.yaml')
F_CSV = os.path.join(RUNS, 'n_150', 'resultados.csv')
Y_CSV = os.path.join(RUNS, 'yolo26n_desde_cero', 'results.csv')
MEJOR = os.path.join(RUNS, 'n_150', 'mejor.pt')
EVAL = os.path.join(RUNS, 'evaluacion.json')
AZUL, NARANJA, GRIS, VERDE = '#1f5aa6', '#d9622b', '#7a7a7a', '#2e8b57'

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.25, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})


def escribir(nombre, texto):
    with open(os.path.join(CAP, nombre), 'w') as f:
        f.write(texto.strip() + '\n')


def pendiente(que):
    return f'\\begin{{nota}}\nPendiente: {que}. Se llena al correr \\cod{{herramientas/generar.py}}.\n\\end{{nota}}'


def leer_csv(ruta):
    with open(ruta) as f:
        filas = list(csv.DictReader(f))
    return {k.strip(): np.array([float(r[k]) for r in filas]) for k in filas[0]}


def copiar_codigo():
    os.makedirs(COD, exist_ok=True)
    for p in glob.glob(os.path.join(PAQUETE, 'cuadrirotor_sim', 'red_fuego', '*.py')):
        if not p.endswith('__init__.py'):
            shutil.copy(p, COD)


def fig_progloss():
    e = np.arange(150)
    w = np.maximum(1 - e / 149, 0) * 0.7 + 0.1
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    ax.plot(e + 1, w, color=AZUL, label='$w$ (uno a muchos)')
    ax.plot(e + 1, 1 - w, color=NARANJA, label='$1-w$ (uno a uno)')
    ax.set_xlabel('época')
    ax.set_ylabel('peso')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, 'progloss.png'))
    plt.close(fig)


def fig_aumentos():
    from cuadrirotor_sim.red_fuego.datos import DatasetYolo
    import random
    random.seed(1)
    np.random.seed(1)
    ds = DatasetYolo(DATOS, 'train', 640, aumentar=True)
    paneles = []
    for i in [0, 500, 3000, 5000, None]:
        if i is None:
            ds.aumentar, i = False, 3000
        img, e = ds[i]
        im = np.ascontiguousarray(img.numpy().transpose(1, 2, 0)[..., ::-1])
        for c, x, y, w, h in e.numpy():
            p = [int(v * 640) for v in (x - w / 2, y - h / 2, x + w / 2, y + h / 2)]
            cv2.rectangle(im, p[:2], p[2:], (0, 0, 255) if c == 0 else (255, 255, 0), 3)
        paneles.append(cv2.resize(im, (400, 400)))
        paneles.append(np.full((400, 8, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(FIG, 'aumentos.png'), np.hstack(paneles[:-1]))


def fig_curvas():
    f = leer_csv(F_CSV)
    y = leer_csv(Y_CSV) if os.path.exists(Y_CSV) else None
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
    axs[0].plot(f['epoca'], f['mAP50-95'], color=AZUL, label='FuegoNet-n')
    axs[1].plot(f['epoca'], f['mAP50'], color=AZUL, label='FuegoNet-n')
    if y is not None:
        axs[0].plot(y['epoch'], y['metrics/mAP50-95(B)'], color=NARANJA, label='YOLO26n (Ultralytics)')
        axs[1].plot(y['epoch'], y['metrics/mAP50(B)'], color=NARANJA, label='YOLO26n (Ultralytics)')
    axs[0].set_title('mAP50-95 (validación)')
    axs[1].set_title('mAP50 (validación)')
    for k, c in (('caja', AZUL), ('clase', NARANJA), ('dist', VERDE)):
        v = f[k] * (10 if k == 'dist' else 1)
        axs[2].plot(f['epoca'], v, color=c, label=k + (' ×10' if k == 'dist' else ''))
    axs[2].set_title('pérdidas FuegoNet (o2o)')
    axs[2].set_yscale('log')
    for a in axs:
        a.set_xlabel('época')
    axs[0].legend(frameon=False, fontsize=7)
    axs[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'curvas.png'))
    plt.close(fig)
    return f, y


def fig_asignacion():
    import torch
    from cuadrirotor_sim.red_fuego.asignador import AsignadorTAL
    from cuadrirotor_sim.red_fuego.datos import a_pixeles, leer_etiquetas, letterbox, listar
    from cuadrirotor_sim.red_fuego.modelo import FuegoNet, anclas, dist_a_caja
    p = torch.load(MEJOR, map_location='cpu')
    m = FuegoNet(p['nc'], p['escala']).eval()
    m.load_state_dict(p['modelo'])
    imgs, _ = listar(DATOS, 'val')
    # una imagen con un fuego mediano
    for ruta in imgs:
        e = leer_etiquetas(ruta)
        f = e[e[:, 0] == 0]
        if len(f) == 1 and 0.05 < f[0, 3] < 0.12:
            break
    img0 = cv2.imread(ruta)
    h, w = img0.shape[:2]
    img, r, (dx, dy) = letterbox(img0, 640)
    x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1)))[None].float() / 255
    m.cabeza.training = True                      # devuelve o2m y o2o sin cambiar BatchNorm
    with torch.no_grad():
        pred = m(x)
    gt = a_pixeles(f, w * r, h * r, dx, dy)
    cajas_gt = torch.tensor(gt, dtype=torch.float32)[None]
    clases = torch.zeros(1, 1, 1)
    valido = torch.ones(1, 1, 1)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
    x1, y1, x2, y2 = gt[0]
    pad = 40
    for k, (ax, (nom, topk, topk2)) in enumerate(zip(axs, [('candidatas', 10, None), ('o2m', 10, None),
                                                            ('o2o', 7, 1)])):
        rama = pred['o2o' if nom == 'o2o' else 'o2m']
        puntos, paso = anclas(rama['mapas'], m.cabeza.pasos)
        cajas = dist_a_caja(rama['dist'], puntos) * paso
        asig = AsignadorTAL(2, topk, topk2, lado_min=16.0)
        _, puntaje, fp = asig(rama['logits'].sigmoid(), cajas, puntos * paso, clases, cajas_gt, valido)
        ax.imshow(img[..., ::-1])
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec='red', lw=1.5))
        c = (puntos * paso).numpy()
        if k == 0:
            dentro = asig._dentro(puntos * paso, cajas_gt)[0, 0].numpy()
            sel = dentro & (paso[:, 0].numpy() == 8)
            ax.scatter(c[sel, 0], c[sel, 1], s=8, c='white', edgecolors='k', linewidths=0.4)
            ax.set_title(f'candidatas en $P_3$: {sel.sum()}')
        else:
            i = fp[0].numpy()
            t = puntaje[0, :, 0].numpy()
            sc = ax.scatter(c[i, 0], c[i, 1], s=18 * (paso[i, 0].numpy() / 8), c=t[i], cmap='viridis',
                            vmin=0, vmax=max(t[i].max(), 1e-3), edgecolors='k', linewidths=0.4)
            ax.set_title(f'{nom}: {i.sum()} positiva' + ('s' if i.sum() != 1 else ''))
            if k == 2:
                b = cajas[0, i].numpy()
                for bb in b:
                    ax.add_patch(plt.Rectangle(bb[:2], *(bb[2:] - bb[:2]), fill=False, ec='yellow',
                                               lw=1, ls='--'))
        ax.set_xlim(x1 - pad, x2 + pad)
        ax.set_ylim(y2 + pad, y1 - pad)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'asignacion.png'))
    plt.close(fig)


def fig_detecciones(ev):
    """Tres imágenes de validación: FuegoNet vs YOLO26n preentrenado (conf 0.25)."""
    from cuadrirotor_sim.red_fuego.datos import listar
    from cuadrirotor_sim.red_fuego.inferir import Detector, dibujar
    from ultralytics import YOLO
    imgs, nombres = listar(DATOS, 'val')
    det = Detector(MEJOR)
    yolo = YOLO(os.path.join(PAQUETE, 'yolo', 'incendio_yolo26n.pt'))
    elegidas = [imgs[42], imgs[170], imgs[330], imgs[520]]
    filas = []
    for ruta in elegidas:
        img = cv2.imread(ruta)
        a = dibujar(img.copy(), det(img, 0.25), det.nombres)
        r = yolo.predict(img, conf=0.25, verbose=False, device=0)[0]
        d = np.concatenate((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None],
                            r.boxes.cls.cpu().numpy()[:, None]), 1)
        b = dibujar(img.copy(), d, det.nombres)
        filas.append(np.hstack((cv2.resize(a, (480, 360)), np.full((360, 6, 3), 255, np.uint8),
                                cv2.resize(b, (480, 360)))))
        filas.append(np.full((6, 966, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(FIG, 'detecciones.png'), np.vstack(filas[:-1]))
    return elegidas


def tex_entrenamiento(f, y):
    i = int(np.argmax(0.1 * f['mAP50'] + 0.9 * f['mAP50-95']))
    t = f['tiempo'][-1] / 60
    txt = f"""
\\section{{Qué pasó en el entrenamiento}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{curvas.png}}
\\caption{{Entrenamiento desde cero. Izquierda y centro: mAP en validación de
FuegoNet-n (este manual) y de YOLO26n de Ultralytics con la misma receta.
Derecha: pérdidas de la cabeza o2o de FuegoNet (escala logarítmica).}}
\\label{{fig:curvas}}
\\end{{figure}}

\\begin{{itemize}}
  \\item El entrenamiento corrió {len(f['epoca'])} épocas en {t:.0f} minutos
        (unos {f['tiempo'][-1] / len(f['epoca']):.0f}~s por época, compartiendo la
        GPU con la corrida de control).
  \\item El mejor modelo (\\cod{{mejor.pt}}) es el de la época {int(f['epoca'][i])}: mAP50 =
        {f['mAP50'][i]:.3f} y mAP50-95 = {f['mAP50-95'][i]:.3f} en validación
        (AP50-95 de fuego {f['AP50-95_fuego'][i]:.3f}, de humo {f['AP50-95_humo'][i]:.3f}).
  \\item En la época 10 ya tenía mAP50 = {f['mAP50'][9]:.3f}: la mayor parte se aprende
        pronto y las últimas épocas afinan las cajas (sube sobre todo el mAP50-95).
  \\item Al apagar el mosaico (últimas 10 épocas) hay un salto pequeño hacia arriba: la
        red se ajusta a imágenes completas como las de vuelo.
\\end{{itemize}}
"""
    if y is not None:
        j = int(np.argmax(0.1 * y['metrics/mAP50(B)'] + 0.9 * y['metrics/mAP50-95(B)']))
        txt += f"""
\\begin{{nota}}
La curva naranja es el control: YOLO26n de Ultralytics, también desde cero, con los
mismos datos y aumentos. Su mejor época ({int(y['epoch'][j])}) da mAP50-95 =
{y['metrics/mAP50-95(B)'][j]:.3f} \\emph{{según su propio validador}}. La comparación
justa, con el mismo código de métricas, está en el capítulo~\\ref{{cap:evaluar}}.
\\end{{nota}}
"""
    escribir('resultados_entrenamiento.tex', txt)


def tex_evaluacion(ev):
    orden = [k for k in ev if k.startswith('FuegoNet')] + \
        [k for k in ev if 'desde_cero' in k or k == 'best.pt'] + [k for k in ev if 'incendio_yolo26n' in k]
    nombre = {'best.pt': 'YOLO26n desde cero', 'incendio_yolo26n.pt': 'YOLO26n preentrenado (COCO)'}
    filas = []
    for k in orden:
        r = ev[k]
        g = f"{r['GFLOPs']:.1f}" if 'GFLOPs' in r else '--'
        filas.append(f"{nombre.get(k, k)} & {r['parametros_M']:.2f} & {g} & {r['P']:.3f} & {r['R']:.3f} & "
                     f"{r['mAP50']:.3f} & {r['mAP50-95']:.3f} & {r['AP50-95_clase'][0]:.3f} & "
                     f"{r['AP50-95_clase'][1]:.3f} & {r['ms_gpu']:.1f} & {r['ms_cpu']:.1f} \\\\")
    fn = ev[orden[0]]
    import torch
    punto = torch.load(MEJOR, map_location='cpu')
    fn_epoca, fn_val = punto['epoca'], punto['metricas']['mAP50-95']
    yc_propio = 0.0
    if os.path.exists(Y_CSV):
        y = leer_csv(Y_CSV)
        yc_propio = y['metrics/mAP50-95(B)'][np.argmax(0.1 * y['metrics/mAP50(B)'] + 0.9 * y['metrics/mAP50-95(B)'])]
    yc = ev.get('best.pt')
    yp = ev.get('incendio_yolo26n.pt')
    txt = f"""
\\section{{Resultados}}

\\begin{{table}}[H]
\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{Las tres redes medidas con el mismo código sobre las 600 imágenes de
validación. P y R con confianza $\\ge0.25$ e IoU 0.5. Mediana de la latencia por imagen (lote 1,
incluye pre y posprocesamiento) en la RTX~5060~Ti y en CPU con 4 hilos.}}
\\label{{tab:comparacion}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lrrrrrrrrrr}}
\\toprule
\\textbf{{Red}} & \\textbf{{Parám.\\ (M)}} & \\textbf{{GFLOPs}} & \\textbf{{P}} & \\textbf{{R}} &
\\textbf{{mAP50}} & \\textbf{{mAP50-95}} & \\textbf{{AP fuego}} & \\textbf{{AP humo}} &
\\textbf{{GPU (ms)}} & \\textbf{{CPU (ms)}}\\\\
\\midrule
{chr(10).join(filas)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    if yc:
        dif = fn['mAP50-95'] - yc['mAP50-95']
        txt += f"""
\\paragraph{{FuegoNet-n contra YOLO26n desde cero.}} Es la comparación que valida
el código: misma arquitectura, mismos datos, mismas épocas y sin preentrenamiento.
Las dos tienen los mismos {fn['parametros_M']:.3f}~M de parámetros: es la misma red.
FuegoNet-n hace {fn['GFLOPs']:.1f}~GFLOPs y no {yc['GFLOPs']:.1f} porque al inferir ya no
calcula la cabeza o2m, que Ultralytics sí calcula aunque no la use. FuegoNet-n queda
{'por encima' if dif > 0 else 'por debajo'} en {abs(dif):.3f} de mAP50-95
({fn['mAP50-95']:.3f} contra {yc['mAP50-95']:.3f}). Es una sola corrida de cada una:
para decir que una es mejor habría que repetir con varias semillas. Lo que sí se
puede concluir es que la red propia \\emph{{reproduce}} a YOLO26n. Además, el
validador de Ultralytics le da {yc_propio:.3f} a su modelo y nuestro código
{yc['mAP50-95']:.3f}, así que también el código de métricas coincide.
"""
    if yp:
        txt += f"""
\\paragraph{{Contra el YOLO26n preentrenado.}} El modelo que usa hoy la simulación da
mAP50-95 = {yp['mAP50-95']:.3f}, {fn['mAP50-95'] - yp['mAP50-95']:.3f} por debajo de
FuegoNet-n. Puede sorprender, porque partió de pesos de COCO (118\\,000 imágenes).
Pero se ajustó solo 60 épocas y FuegoNet entrenó 150, y las imágenes de validación
vienen del mismo simulador que las de entrenamiento. En este caso, más épocas sobre
datos del dominio pesan más que el preentrenamiento. Con imágenes reales de
incendios, donde COCO sí aporta, el resultado podría invertirse
(capítulo~\\ref{{cap:siguientes}}). En mAP50 las tres redes empatan
({fn['mAP50']:.3f}): todas encuentran el fuego. La diferencia está en qué tan
ajustadas quedan las cajas.

\\paragraph{{Velocidad.}} En la GPU las tres tardan entre {min(r['ms_gpu'] for r in ev.values()):.0f}
y {max(r['ms_gpu'] for r in ev.values()):.0f}~ms por imagen (mediana de 100), contando el
pre y el posprocesamiento: unas 250 imágenes por segundo. En CPU, FuegoNet-n tarda
{fn['ms_cpu']:.0f}~ms y los YOLO26n entre {min(yc['ms_cpu'], yp['ms_cpu']):.0f} y
{max(yc['ms_cpu'], yp['ms_cpu']):.0f}~ms. Aunque es la misma red, Ultralytics infiere con
la imagen \\emph{{rectangular}} de $640\\times480$ y FuegoNet con el cuadrado de
$640\\times640$ (25\\,\\% más píxeles). Con \\cod{{Detector(..., rect=True)}} FuegoNet
baja a unos 23~ms, pero su mAP50-95 cae de 0.899 a 0.886 porque se entrenó solo con
imágenes cuadradas. Por eso, por omisión, se usa el cuadrado. Los tiempos en CPU
varían algunos milisegundos de una medición a otra.

\\begin{{leccion}}
Durante el entrenamiento, el mejor modelo (época {fn_epoca}) marcaba mAP50-95 =
{fn_val:.3f} y aquí da {fn['mAP50-95']:.3f}. La diferencia viene de la precisión:
\\cod{{entrenar.py}} valida con \\cod{{bfloat16}} (rápido) y \\cod{{evaluar.py}} en
\\cod{{float32}}. Con 8 bits de mantisa, una coordenada entre 512 y 1024 píxeles se
redondea de 4 en 4 píxeles, y eso castiga los umbrales de IoU altos (0.90, 0.95).
Se comprobó validando el mismo \\cod{{mejor.pt}} de las dos formas: 0.880 en
\\cod{{bfloat16}} y 0.898 en \\cod{{float32}}. Para reportar resultados,
usa siempre \\cod{{evaluar.py}}. Queda pendiente decodificar las cajas en
\\cod{{float32}} dentro de \\cod{{Cabeza.decodificar}}: así la elección del mejor
modelo también usaría la métrica exacta.
\\end{{leccion}}
"""
    txt += """
\\begin{figure}[H]
\\centering
\\includegraphics[width=0.92\\textwidth]{detecciones.png}
\\caption{Detecciones con confianza $\\ge 0.25$ en cuatro imágenes de validación.
Izquierda: FuegoNet-n (sin NMS). Derecha: YOLO26n preentrenado de la simulación.}
\\label{fig:detecciones}
\\end{figure}
"""
    escribir('resultados_evaluacion.tex', txt)


def tex_inferencia():
    from cuadrirotor_sim.red_fuego.datos import listar
    imgs, _ = listar(DATOS, 'val')
    out = subprocess.run([sys.executable, '-m', 'cuadrirotor_sim.red_fuego.inferir', MEJOR, imgs[42],
                          os.path.join(RUNS, 'deteccion_ejemplo.jpg')], cwd=PAQUETE,
                         capture_output=True, text=True).stdout.strip()
    escribir('resultados_inferencia.tex', f"""
\\begin{{resultado}}
\\begin{{lstlisting}}
{out}
\\end{{lstlisting}}
Cada fila es una caja: clase, confianza y $(x_1, y_1, x_2, y_2)$ en píxeles de la imagen
original de $640\\times480$.
\\end{{resultado}}
""")


def tex_onnx():
    from cuadrirotor_sim.red_fuego.datos import listar
    destino = os.path.join(RUNS, 'fuegonet_n.onnx')
    foto = listar(DATOS, 'val')[0][42]
    out = subprocess.run([sys.executable, '-m', 'cuadrirotor_sim.red_fuego.exportar', MEJOR, destino, foto],
                         cwd=PAQUETE, capture_output=True, text=True).stdout.strip()
    mb = os.path.getsize(destino) / 1e6 if os.path.exists(destino) else 0
    escribir('resultados_onnx.tex', f"""
\\begin{{resultado}}
\\begin{{lstlisting}}
{out}
\\end{{lstlisting}}
El archivo pesa {mb:.1f}~MB. La entrada se llama \\cod{{imagen}} $(1,3,640,640)$ y la
salida \\cod{{detecciones}} $(1,300,6)$.
\\end{{resultado}}
""")


def main():
    for d in (FIG, CAP):
        os.makedirs(d, exist_ok=True)
    copiar_codigo()
    fig_progloss()
    fig_aumentos()
    f = y = None
    if os.path.exists(F_CSV):
        f, y = fig_curvas()
        tex_entrenamiento(f, y)
    else:
        escribir('resultados_entrenamiento.tex', pendiente('curvas del entrenamiento de 150 épocas'))
    if os.path.exists(MEJOR):
        fig_asignacion()
        tex_inferencia()
        tex_onnx()
    else:
        for n in ('resultados_inferencia.tex', 'resultados_onnx.tex'):
            escribir(n, pendiente('salida con el modelo entrenado'))
    if os.path.exists(EVAL):
        ev = json.load(open(EVAL))
        fig_detecciones(ev)
        tex_evaluacion(ev)
    else:
        escribir('resultados_evaluacion.tex', pendiente('tabla comparativa de las tres redes'))
    print('Listo:', MANUAL)


if __name__ == '__main__':
    main()
