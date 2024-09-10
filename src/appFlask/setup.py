import setuptools

setuptools.setup(
    name='saison',
    author='la team Avengers',
    description='for later',
    packages=setuptools.find_packages(),
    install_requires=[
        "opencv-python==4.10.0.84"
        "Flask==3.0.3"
        "werkzeug==3.0.4"
        "ultralytics==8.2.90"
    ],
)