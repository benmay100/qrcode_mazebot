from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'qrcode_mazebot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'assets'), glob('assets/*.png'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ben-may',
    maintainer_email='benalexandermay@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "qr_code_maze_driver_v1 = qrcode_mazebot.qr_code_maze_driver_v1:main",
            "qr_code_maze_driver_v2 = qrcode_mazebot.qr_code_maze_driver_v2:main",
            "qr_code_maze_driver_v3 = qrcode_mazebot.qr_code_maze_driver_v3:main"
        ],
    },
)
