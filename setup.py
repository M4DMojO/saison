import setuptools
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('src/__init__.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


setuptools.setup(
    name='saison',
    version=main_ns['__version__'],
    author='la team joeffrey',
    description='for later',
    packages=setuptools.find_packages(),
    install_requires=[
        "-e .",
        "click==7.0", 
        "Pillow==9.1.1"
    ],
    entry_points='''
    [console_scripts]
    saison=src.cli:saison
    '''
)