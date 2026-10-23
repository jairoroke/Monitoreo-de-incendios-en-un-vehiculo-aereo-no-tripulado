from glob import glob
import os

from setuptools import find_packages, setup

paquete = 'cuadrirotor_sim'

setup(
    name=paquete,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + paquete]),
        ('share/' + paquete, ['package.xml']),
        ('share/' + paquete + '/launch', glob('launch/*.py')),
        ('share/' + paquete + '/config', glob('config/*.yaml') + glob('config/*.config')),
        ('share/' + paquete + '/config/ardupilot', glob('config/ardupilot/*.parm')),
        ('share/' + paquete + '/yolo', glob('yolo/*.pt')),
        ('share/' + paquete + '/worlds', glob('worlds/*.sdf')),
    ] + [(os.path.join('share', paquete, d), [os.path.join(d, f) for f in fs])
         for d, _, fs in os.walk('models') if fs],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jairo Olguín Roque',
    maintainer_email='jairoolguin1@gmail.com',
    description='Simulación de un cuadrirrotor en Gazebo Harmonic.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controlador_pd = cuadrirotor_sim.controlador_pd:main',
            'controlador_sta = cuadrirotor_sim.controlador_sta:main',
            'viento = cuadrirotor_sim.viento:main',
            'mision_ardupilot = cuadrirotor_sim.mision_ardupilot:main',
            'busqueda_incendio = cuadrirotor_sim.busqueda_incendio:main',
            'fauna = cuadrirotor_sim.fauna:main',
        ],
    },
)
