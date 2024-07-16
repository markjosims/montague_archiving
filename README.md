# montague_archiving
ASR scripts for the Montague biography project

Main file is `annotate.py`

usage: Annotation runner [-h] [-i INPUT] [-s {asr-only,asr-first,drz-first,multitier}]
    [-n NUM_SPEAKERS] [-m ASR_MODEL] [-d DRZ_MODEL] [-D DEVICE] [-c CHUNK_LENGTH_S] 
    [-w] [-r] [-b ASR_BATCH_SIZE]

optional arguments:
  -h, --help
    show this help message and exit
  -i INPUT, --input INPUT
    .wav file or directory of .wav files to annotate
  -s {asr-only,asr-first,drz-first,multitier}, --strategy {asr-only,asr-first,drz-first,multitier}
    Specify what pipeline to use for annotation. `asr-first` (default) will run Whisper first then
    diarization with PyAnnote, and the PyAnnote diarization will be used to decide the speaker identityfor
    each chunk output by Whisper. `asr-only` will run Whisper without performing speaker diarization.
    `drz-first` will run diarization with PyAnnote first and then send each speaker turn to Whisper for
    transcription. This will likely result in poor turn boundaries, so `asr-first` should generally be
    preferred. `multitier` will output an .eaf file with Whisper annotations and PyAnnote speaker turns on
    separate tiers, and is mostly useful for debugging and understanding the decisions that ASR and
    diarization are making independently.
  -n NUM_SPEAKERS, --num_speakers NUM_SPEAKERS
    Number of speakers in file. Default 2.
  -m ASR_MODEL, --asr_model ASR_MODEL
    ASR model path. Default is openai/whisper-medium.en.
  -d DRZ_MODEL, --drz_model DRZ_MODEL
    DRZ model path. Default is pyannote/speaker-diarization-3.1.
  -D DEVICE, --device DEVICE
    Device to run GPU on. Defaults to `cuda:0` if available else `cpu`.
  -c CHUNK_LENGTH_S, --chunk_length_s CHUNK_LENGTH_S
    Chunk size to input to Whisper pipeline (default 30s). Note that this will not affect the size of
    chunks OUTPUT by Whisper, so this should only be modified if there is not enough memory to process 30s
    at a time.
  -w, --return_word_timestamps
    Whisper by default chunks speech more or less into utterances. Use this option to chunk by word, which
    may give more precise time accuracy When detecting speaker changes. (I recommend using this option.)
  -r, --recursive
    If input is a directory, will search for files .wav files recursively
  -b ASR_BATCH_SIZE, --asr_batch_size ASR_BATCH_SIZE
    Inference batch size for ASR. Default 8.

For each input .wav file, saves a .eaf and .txt file with annotations for that
recording with the same path as the input file.