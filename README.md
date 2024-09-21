# The Playlister
The Playlister is an audio processing and deep learning project. It tackles two main challenges:

* Transition point selection - Selecting the best transition point between two consecutive songs in a playlist.
* Songs listing - Optimizing the order of songs to ensure smooth transitions.

_View "Playlister.pdf" for project summary and docummentation._


## Quick Introduction:
Music generative model creates new pieces of music autonomously. As part of the model’s flow, it calculates the conditional probabilities (given some piece of audio). This project main idea is to extract these probabilities and use them for songs listing and transition points selection. We use the DTW algorithm as baseline solution for the same problems.


## Setup
We used an NVIDIA GPU running on Linux with Driver Version 525.116 and CUDA Version 10.0. The GPU was an NVIDIA TITAN Xp with 12 GB of graphic memory, so this setup or more advanced versions should work.

**Requirements**
- Python version: 3.10
- Required packages are at requirements.txt file.
- _You can create a conda virtual environment and install the requirements by running the following commands:_
```bash
conda create -n playlister python=3.10
conda activate playlister
pip install -r requirements.txt
```

## Creating a playlist using musicgen:
Run with single GPU

`python -u musicgen_based_playlist_generator.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files> --outpath <Directory path for saving the results> --fade_duration <seconds (integer)>`


## Creating a playlist using DTW (baseline):
Run on CPU

`python -u baseline_dtw_playlist_generator.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files> --outpath <Directory path for saving the results> --fade_duration <seconds (integer)>`


## Other tools:
### Characterize hyper params to find the best fit:
Run on single GPU

`python -u find_hyper_params.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files>`

### Saves plot represents evaluations based on tempo metrics (saves result to current directory):
Run on CPU

`python evaluate_using_tempo.py`

