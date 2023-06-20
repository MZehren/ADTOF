### Update
Clear mapping from different datasets (e.g., [Slakh](https://github.com/ethman/slakh-generation)) to General MIDI has been added in the file [instrumentsMapping.py](./adtof/ressources/instrumentsMapping.py)

# ADTOF
This repository contains additional material for the paper **ADTOF: A large dataset of non-synthetic music for automatic drum transcription** by Mickaël Zehren, Marco Alunno, and Paolo Bientinesi.

## Abstract
The state-of-the-art methods for drum transcription in the presence of melodic instruments (DTM) are machine learning models trained in a supervised manner, which means that they rely on labeled datasets. The problem is that the available public datasets are limited either in size or in realism, and are thus suboptimal for training purposes. Indeed, the best results are currently obtained via a rather convoluted multi-step training process that involves both real and synthetic datasets. To address this issue, starting from the observation that the communities of rhythm games players provide a large amount of annotated data, we curated a new dataset of crowdsourced drum transcriptions. This dataset contains real-world music, is manually annotated, and is about two orders of magnitude larger than any other non-synthetic dataset, making it a prime candidate for training purposes. However, due to crowdsourcing, the initial annotations contain mistakes. We discuss how the quality of the dataset can be improved by automatically correcting different types of mistakes. When used to train a popular DTM model, the dataset yields a performance that matches that of the state-of-the-art for DTM, thus demonstrating the quality of the annotations.


## Installation
To build the dataset or use the pre-trained ADT models, you first need to install the scripts shared in this repository. This can be done with the script [setup.py](./setup.py) with the following command line:

    pip3 install .

:warning: This repository has been tested on macOS Catalina with **Python 3.8** and **pip 21.1**.


## Dataset
A copy of the `ADTOF` dataset is available in the folder [/dataset](/dataset). This copy is shared without the audio files containing copyrighted content. The dataset has the following structure:
- [/annotations/aligned_beats/](./dataset/annotations/aligned_beats/): text files with the time corrected beats location
- [/annotations/aligned_drum/](./dataset/annotations/aligned_drum/): text files with the time corrected drum onsets location. 
- [/annotations/[raw/converted/aligned]_midi/](./dataset/annotations/): contains intermediate files kept for debuging purposes
- [/annotations/manual_substraction](./dataset/annotations/manual_substraction): list of files manually removed from the dataset 
- [/audio/audio/](./dataset/audio/audio): audio files 
- [/estimations/beats/](./dataset/estimations/beats): beats estimated by [madmom](https://github.com/CPJKU/madmom)
- [/estimations/beats_activation/](./dataset/estimations/beats_activation): activation output of madmom's beats estimation model

All the text files in the dataset are in a [tab-separated values](https://en.wikipedia.org/wiki/Tab-separated_values) format. The drum instruments are named according to the [standard midi](https://en.wikipedia.org/wiki/General_MIDI#Percussive) pitches like so:
 - 35: Bass drum
 - 38: Snare drum 
 - 47: Tom
 - 42: Hi-hat
 - 49: Crash and ride cymbal

You can build your copy of the dataset (with audio files) by following the next three steps.

### 1. Download custom charts
To build the dataset you need to download custom charts. We created the script [/bin/downloadRhythmGamingWorld.py](/bin/downloadRhythmGamingWorld.py) which downloads automatically charts from the website [Rhythm Gaming World](https://rhythmgamingworld.com/). :warning: we are not affiliated with Rhythm Gaming World and we do not control copyrighted material being uploaded. Use the following script at your own risk.

    downloadRhythmGamingWorld.py [-h] outputFolder


    Download custom charts fron the website https://rhythmgamingworld.com/
    
    positional arguments:
        outputFolder  Path to the destination folder where the files are downloaded.
    
    optional arguments:
        -h, --help    show this help message and exit


### 2. Convert the custom charts to the `PhaseShift` file format
The charts you downloaded in the previous step are in different formats meant for different rhythm games. But the automatic cleansing requires that the charts downloaded are following specifically the `PhaseShift` file format (i.e. a folder containing a *song.ogg* and a *notes.mid* file). You can easily convert the charts downloaded to the good file format with the software [C3 CON Tools](https://rhythmgamingworld.com/forums/topic/c3-con-tools-v401-8142020-weve-only-just-begun/) (tested on Windows 10). After downloading and lauching C3 CON Tools, the conversion is done on the graphical user interface following this procedure:
1. Click on **Phase Shift Converter**
2. Click on **Change Input Folder** and select the folder containing the custom charts previous downloaded
3. Click on **Begin**

### 3. Automatic grooming
The custom charts you dowloaded and transformed into the `PhaseShift` file format can now be converted into a usable dataset (similar to the copy [/dataset](/dataset)) with the script [/bin/automaticGrooming.py](/bin/automaticGrooming.py):

    automaticGrooming.py [-h] [-p] inputFolder outputFolder


    Process a chart folder with the automatic cleaning procedure

    positional arguments:
    inputFolder     Path to the chart folder.
    outputFolder    Path to the destination folder of the dataset.

    optional arguments:
    -h, --help      show this help message and exit
    -p, --parallel  Set to run the cleaning in parallel


## Models
You can access the pre-trained models we evaluated in the folder [/adtof/models/](./adtof/models). You can use the models directly to transcribe audio tracks with the script [/bin/drumTranscriptor.py](/bin/drumTranscriptor.py):

    drumTranscriptor.py [-h] [-m MODEL] inputPath outputPath


    Use one of trained model to perform ADT
    
    positional arguments:
        inputPath             Path to a music file or folder containing music
        outputPath            Path to output folder
    
    optional arguments:
        -h, --help            show this help message and exit
        -m MODEL, --model MODEL
                            Name of the pre-trained model used for the transcription. Values: crnn-ADTOF,
                            crnn-all, crnn-ptTMIDT. (default: crnn-ADTOF)

## Raw results
The raw results of the cross-validation are shared in the folder [/evaluation](./evaluation). The plots available in the paper are created with the script [/bin/plotAlgorithmAccuracy.py](/bin/plotAlgorithmAccuracy.py).

