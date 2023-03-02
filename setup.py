import setuptools
from packagename.version import Version


setuptools.setup(name='anomaly-detection',
                 version=Version('1.0.0').number,
                 description='SynCAN anomaly detection package',
                 long_description=open('README.md').read().strip(),
                 author='Brendon Felix',
                 author_email='brendon.felix.c@gmail.com',
#                  url='http://path-to-my-packagename',
                 py_modules=['anomalydetection'],
#                  install_requires=[],
#                  license='MIT License',
#                  zip_safe=False,
#                  keywords='boilerplate package',
                 classifiers=['Packages', 'AnomalyDetection'])
