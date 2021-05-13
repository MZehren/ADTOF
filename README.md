# ADTOF
This repository contains additional material for the paper **ADTOF: A large dataset of real western music annotated for automatic drum transcription** by anonymous.

## Abstract
The state-of-the-art methods for Automatic Drum Transcription (ADT) are machine learning models trained in a supervised manner, which means that they rely on labeled datasets. The problem is that the available public datasets are limited either in size or in realism, and are thus suboptimal for training purposes. Indeed, the best results are currently obtained via a rather convoluted multi-step training process that involves both real and synthetic datasets. To address this issue, starting from the observation that the communities of rhythm games players provide a large amount of annotated data, we curated a new dataset of crowdsourced drum transcriptions. This dataset contains real-world music, is manually annotated, and is about two orders of magnitude larger than any other non-synthetic dataset, making it a prime candidate for training purposes. However, due to crowdsourcing, the initial annotations contain mistakes. We discuss how the quality of the dataset can be improved by automatically correcting different types of mistakes. When used to train a popular ADT model, the dataset yields a performance that matches that of the state-of-the-art for ADT, thus demonstrating the quality of the annotations.

## Installation
To build the dataset or use the pre-trained ADT models, you will first need to install the scripts shared in this repository. This can be done with the [setup.py](./setup.py) script with the following command line:
> pip3 install .

| :warning: This repository has been tested on macOS Catalina with **Python 3.8** and **pip 21.1** |
| ------------------------------------------------------------------------------------------------ |

## Dataset
A copy of the `ADTOF` dataset is available in the [/dataset](/dataset) without the audio files containing copyrighted content. The dataset has the following structure:
- [/annotations/aligned_beats/](./dataset/annotations/aligned_beats/): text files with the time corrected beats location
- [/annotations/aligned_drum/](./dataset/annotations/aligned_drum/): text files with the time corrected drum onsets location
- [/annotations/aligned_drum/[raw/converted/aligned]_midi/](./dataset/annotations/): contains intermediate files kept for debuging purposes
- [/annotations/aligned_drum/manual_substraction](./dataset/annotations/manual_substraction): list of files removed from the dataset 
- [/audio/audio/](./dataset/audio/audio): audio files 
- [/estimations/beats/](./dataset/estimations/beats): beats estimated by [madmom](https://github.com/CPJKU/madmom)
- [/estimations/beats_activation/](./dataset/estimations/beats_activation): activation output of madmom's beats estimation model

You can build th 
### 1. Download Custom charts
To build the dataset you will first need custom charts. The following script allows to download charts from the website [Rhythm Gaming World](https://rhythmgamingworld.com/).
TODO: add the 2x bass versions
>usage: downloadRhythmGamingWorld.py [-h] outputFolder
>
>Download custom charts fron the website https://rhythmgamingworld.com/
>
>positional arguments:
>  outputFolder  Path to the destination folder where the files are downloaded.
>
>optional arguments:
>  -h, --help    show this help message and exit


Rhythm Gaming World lists custom charts for multiple games, but `ADTOF` requires that the charts downloaded are following the `PhaseShift` file format (i.e. a folder containing a *song.ogg* and a *notes.mid* file). The charts downloaded in other games' file format can be automatically converted with the software [C3 CON Tools](https://rhythmgamingworld.com/forums/topic/c3-con-tools-v401-8142020-weve-only-just-begun/) (tested on Windows 10). The conversion is done with a graphical user interface following the procedure:
1. Click on **Phase Shift Converter**
2. Click on **Change Input Folder** and select the folder containing the custom charts to convert
3. Click on **Begin**


### 2. Cleaning
TODO HANDLE LOGS
>usage: buildDataset.py [-h] [-p] inputFolder outputFolder
>
>Process a chart folder with the automatic cleaning
>
>positional arguments:
>  inputFolder     Path to the chart folder.
>  outputFolder    Path to the destination folder.
>
>optional arguments:
>  -h, --help      show this help message and exit
>  -p, --parallel  Set to run the cleaning in parallel


## Models
Trained models are available in the [/adtof/models](./adtof/models) folder. You can use them directly with the [drumTranscriptor](/bin/drumTranscriptor.py) script:
>drumTranscriptor.py [-h] inputPath outputPath model
>
>todo
>
>positional arguments:
>  inputPath   Path to music or folder containing music to transcribe
>  outputPath  Path to output folder
>  model       name of the nodel to train, possible choice
>
>optional arguments:
>  -h, --help  show this help message and exit

## Raw results
The folder [/evaluatuion](./evaluation) contains the raw results of the cross validation. The plots in the paper are created with the script [plotAlgorithmAccuracy](/bin/plotAlgorithmAccuracy.py)

## License
TODO