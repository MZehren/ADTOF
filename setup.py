import setuptools
import glob

requirements = ["mido", "librosa", "pyunpack", "patool", "jellyfish", "pydot", "graphviz", "pyguitarpro"]

with open("README.rst", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='ADTOF',
    version='0.1',
    scripts=glob.glob('bin/*'),
    author="Mickael Zehren",
    author_email="mickael.zehren@gmail.com",
    description="Automatic drums transcription database conversion",
    long_description=long_description,
    url="https://github.com/MZehren/ADTOF",
    packages=setuptools.find_packages(),
    package_data={"adtof": ['converters/mappingDictionaries/*']},
    install_requires=requirements,
    classifiers=[
        # "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)
