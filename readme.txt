Creating a playlist using musicgen:
command line running example (run with single GPU):
python -u musicgen_based_playlist_generator.py --home_dir /home/joberant/NLP_2324/yaelshemesh --songs_dir /home/joberant/NLP_2324/yaelshemesh/yael_playlist --outpath /home/joberant/NLP_2324/yaelshemesh/playlister_project/outputs --fade_duration 2


Creating a playlist using DTW (baseline)
command line running example (run on CPU)
python -u baseline_dtw_playlist_generator.py --home_dir /home/joberant/NLP_2324/yaelshemesh --songs_dir /home/joberant/NLP_2324/yaelshemesh/yael_playlist --outpath /home/joberant/NLP_2324/yaelshemesh/playlister_project/outputs --fade_duration 2


The directory "ofir_playlist" contains 10 songs used for tests
The directory "yael_playlist" contains another 10 songs
The directory "outputs" contains the playlist created using musicgen and the playlist created using DTW (both with and without fader)


Other scripts:
Characterize hyper params to find the best fit:
command line running example (run on single GPU)
python -u find_hyper_params.py --home_dir /home/joberant/NLP_2324/yaelshemesh --songs_dir /home/joberant/NLP_2324/yaelshemesh/yael_playlist

Saves plot represents evaluations based on tempo metrics (saves result to current directory):
command line running example (run on CPU)
python evaluate_using_tempo.py

