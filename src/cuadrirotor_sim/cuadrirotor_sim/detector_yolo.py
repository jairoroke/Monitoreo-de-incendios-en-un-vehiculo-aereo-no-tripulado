"""Detector de incendios con YOLO26 sobre la cámara del dron.

Lee /cuadrirotor/camara, corre el modelo (entrenado con scripts/entrenar_yolo.py
sobre imágenes del simulador, clases 'fuego' y 'humo') y publica:
  /cuadrirotor/detecciones   vision_msgs/Detection2DArray, con la marca de tiempo
                             de la imagen, para emparejarla con la pose del dron.
  /cuadrirotor/camara_yolo   la imagen con las cajas dibujadas (se ve en el panel
                             "Detección YOLO" de Gazebo).

Se ejecuta con el Python de ~/.venvs/yolo (ultralytics, PyTorch con CUDA), que
también ve los paquetes de ROS 2.
"""
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

COLORES = {'fuego': (255, 60, 0), 'humo': (0, 200, 255)}


class DetectorYolo(Node):

    def __init__(self):
        super().__init__('detector_yolo')
        p = self.declare_parameters('', [
            ('modelo', ''), ('confianza', 0.35), ('periodo', 0.1), ('dispositivo', '0'),
        ])
        v = {x.name: x.value for x in p}
        if not os.path.exists(v['modelo']):
            raise RuntimeError(f"No existe el modelo YOLO: {v['modelo']!r}")
        self.modelo = YOLO(v['modelo'])
        self.conf, self.disp = v['confianza'], v['dispositivo']
        self.ultima = None
        self.create_subscription(Image, '/cuadrirotor/camara', self._imagen, 2)
        self.pub = self.create_publisher(Detection2DArray, '/cuadrirotor/detecciones', 10)
        self.pub_img = self.create_publisher(Image, '/cuadrirotor/camara_yolo', 2)
        # Calentar el modelo para que la primera imagen no tarde.
        self.modelo.predict(np.zeros((480, 640, 3), np.uint8), device=self.disp, verbose=False)
        self.create_timer(v['periodo'], self._procesar)
        self.get_logger().info(f"YOLO listo: {os.path.basename(v['modelo'])}, clases {self.modelo.names}")

    def _imagen(self, msg):
        self.ultima = msg

    def _procesar(self):
        msg, self.ultima = self.ultima, None
        if msg is None:
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        t0 = time.perf_counter()
        r = self.modelo.predict(img[..., ::-1], conf=self.conf, device=self.disp, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        salida = Detection2DArray(header=msg.header)
        dibujo = img.copy()
        for caja, clase, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy()):
            nombre = self.modelo.names[int(clase)]
            x0, y0, x1, y1 = caja
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x, d.bbox.center.position.y = float(x0 + x1) / 2, float(y0 + y1) / 2
            d.bbox.size_x, d.bbox.size_y = float(x1 - x0), float(y1 - y0)
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = nombre, float(conf)
            d.results.append(h)
            salida.detections.append(d)
            c = COLORES.get(nombre, (255, 255, 255))
            cv2.rectangle(dibujo, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
            cv2.putText(dibujo, f'{nombre} {conf:.2f}', (int(x0), max(14, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        cv2.putText(dibujo, f'YOLO26  {ms:.0f} ms', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        self.pub.publish(salida)
        out = Image(header=msg.header, height=msg.height, width=msg.width, encoding='rgb8',
                    step=msg.width * 3, data=dibujo.tobytes())
        self.pub_img.publish(out)


def main():
    rclpy.init()
    nodo = DetectorYolo()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
