import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADTOF",
    version="1.0",
    scripts=glob.glob("bin/*"),
    author="anonymous",
    author_email="anonymous",
    description="Additional material for the paper 'ADTOF: A large dataset of real western music annotated for automatic drum transcription by anonymous.'",
    long_description=long_description,
    url="https://github.com/anonymous",
    packages=setuptools.find_packages(),
    install_requires=[
        "librosa",
        "madmom",
        "sklearn",
        "tensorflow",
        "matplotlib",
        "pandas",
        "mir_eval",
        "jellyfish",
        "pyunpack",
        "ffmpeg",
        "pretty_midi",
        "beautifulsoup4",
        "tapcorrect @ git+https://github.com/MZehren/tapcorrect#subdirectory=python&egg=tapcorrect",
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
)
