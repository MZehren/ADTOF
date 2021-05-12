# ADTOF

This repository contains additional material for the paper **ADTOF: A large dataset of real western music annotated for automatic drum transcription** by anonymous.

## Abstract
The state-of-the-art methods for Automatic Drum Transcription (ADT) are machine learning models trained in a supervised manner, which means that they rely on labeled datasets. The problem is that the available public datasets are limited either in size or in realism, and are thus suboptimal for training purposes. Indeed, the best results are currently obtained via a rather convoluted multi-step training process that involves both real and synthetic datasets. To address this issue, starting from the observation that the communities of rhythm games players provide a large amount of annotated data, we curated a new dataset of crowdsourced drum transcriptions. This dataset contains real-world music, is manually annotated, and is about two orders of magnitude larger than any other non-synthetic dataset, making it a prime candidate for training purposes. However, due to crowdsourcing, the initial annotations contain mistakes. We discuss how the quality of the dataset can be improved by automatically correcting different types of mistakes. When used to train a popular ADT model, the dataset yields a performance that matches that of the state-of-the-art for ADT, thus demonstrating the quality of the annotations.

## Installation

To build the dataset or use the pre-trained ADT models, you will first need to install the scripts shared in this repository. This can be done with the [setup.py](./setup.py) script with the following command line:
> pip3 install -e .

TODO: add the version of the other scripts
| :warning: This repository has been tested on macOS Catalina with **Python 3.8** and **pip 20.3** |
| ------------------------------------------------------------------------------------------------ |

## Build the dataset
### 1. Download Custom charts

To build the dataset you will first need custom charts. The following script allows to download charts from the website [rhythmgamingworld](https://rhythmgamingworld.com/).
>downloadRhythmGamingWorld.py [-h] outputFolder
>
>positional arguments:
>  outputFolder  Path to the destination folder.

After to this step, you might want to ensure that all the tracks downloaded are following the `PhaseShift` file format, which is required by ADTOF. The charts downloaded in the wrong file format can be done automatically with the software [C3 CON Tools](https://rhythmgamingworld.com/forums/topic/c3-con-tools-v401-8142020-weve-only-just-begun/) and using the GUI by clicking on the button `Phase Shift Converter` and then 




### Cleaning

### Transcribe

## License
