import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADTOF",
    version="1.0",
    scripts=glob.glob("bin/*"),
    author="anonyme",
    author_email="anonyme",
    description="Automatic drums transcription database conversion",
    long_description=long_description,
    url="https://github.com/MZehren/ADTOF",
    packages=setuptools.find_packages(),
    # package_data={"adtof": ["converters/mappingDictionaries/*"]},
    install_requires=[
        "librosa",
        "cython",
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
        "tapcorrect @ git+https://github.com/MZehren/tapcorrect#subdirectory=python&egg=tapcorrect"
    ],
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
)
