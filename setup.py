import setuptools
from packagename.version import Version


setuptools.setup(name='anomaly-detection',
                 version='1.0',
                 description='SynCAN anomaly detection package',
                 long_description=open('README.md').read().strip(),
                 author='Brendon Felix',
                 author_email='brendon.felix.c@gmail.com',
                 packages=['anomaly-detection'],
                 install_requires=[])
