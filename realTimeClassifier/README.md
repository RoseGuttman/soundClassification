## Classifies sound recorded in real-time to one of the classes from the audioset.

## Installation
* Install system packages
```bash
sudo apt-get install libportaudio2 portaudio19-dev
```
* Install python requirements
```bash
pip install -r requirements.txt
```

* Put trained models in the models directory

## Running
#### To process prerecorded wav file
run
```bash
python parse_file.py path_to_your_file.wav
```
_Note: file should have 16000 rate_

#### To capture and process audio from mic
run
```bash
python capture.py
```
It will capture and process samples in a loop.\
To get info about parameters run
```bash
python capture.py --help
```

#### To start web server
run
```bash
python daemon.py
```
By default you can reach it on http://127.0.0.1:8000 \
It will:
* Capture data form your mic
* Process data
* Send predictions to web interface
