import setuptools

setuptools.setup(
    name='saison',
    author='la team Avengers',
    description='for later',
    packages=setuptools.find_packages(),
    version='1.0.0',
    install_requires=[
        "opencv-python==4.10.0.84"
        "Flask==3.0.3"
        "ultralytics==8.2.90"
        "tensorflow == 2.17.0"
        "google-clood == 0.34.0"
        "google-cloud-storage==2.18.2"
    ],
)