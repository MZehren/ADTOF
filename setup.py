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
    description="Additional material for the paper 'ADTOF: A large dataset of non-synthetic music for automatic drum transcription'",
    long_description=long_description,
    url="https://github.com/anonymous",
    packages=setuptools.find_packages(),
    package_data={"adtof": ["models/*"]},
    install_requires=[
        "librosa>=0.8.0",
        "madmom>=0.16.1",
        "scikit-learn>=0.24.1",
        "tensorflow>=2.4.1",
        "matplotlib>=3.4.2",
        "pandas>=1.2.4",
        "mir_eval>=0.6",
        "jellyfish",
        "pyunpack>=0.2.2",
        "ffmpeg-python",
        "pretty_midi>=0.2.9",
        "beautifulsoup4",
        "tapcorrect @ git+https://github.com/MZehren/tapcorrect#subdirectory=python&egg=tapcorrect",
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
)
