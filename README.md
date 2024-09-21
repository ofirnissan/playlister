## Creating a playlist using musicgen:
Run with single GPU

`python -u musicgen_based_playlist_generator.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files> --outpath <Directory path for saving the results> --fade_duration 2`


## Creating a playlist using DTW (baseline):
Run on CPU

`python -u baseline_dtw_playlist_generator.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files> --outpath <Directory path for saving the results> --fade_duration 2`


## Other tools:
### Characterize hyper params to find the best fit:
Run on single GPU

`python -u find_hyper_params.py --home_dir <YOUR HOME DIR> --songs_dir <Directory path with mp3 files>`

### Saves plot represents evaluations based on tempo metrics (saves result to current directory):
Run on CPU

`python evaluate_using_tempo.py`

