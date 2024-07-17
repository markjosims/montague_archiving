from annotate import annotate, DIARIZE_URI, DEVICE
from typing import Optional, Sequence
from gooey import Gooey, GooeyParser
import torch

ASR_CHOICES = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3",
]

def init_parser() -> GooeyParser:
    parser = GooeyParser()
    parser.add_argument(
        "-i",
        "--input",
        help=".wav file or directory of .wav files to annotate",
        widget='DirChooser',
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=["asr-only", "asr-first", "drz-first", "multitier"],
        default="asr-first",
        help="Specify what pipeline to use for annotation. "\
        +"`asr-first` (default) will run Whisper first then diarization with PyAnnote, "\
        +"and the PyAnnote diarization will be used to decide the speaker identity"\
        +"for each chunk output by Whisper. "\
        +"`asr-only` will run Whisper without performing speaker diarization. "\
        +"`drz-first` will run diarization with PyAnnote first and then send each "\
        +"speaker turn to Whisper for transcription. This will likely result in poor "\
        +"turn boundaries, so `asr-first` should generally be preferred. "\
        +"`multitier` will output an .eaf file with Whisper annotations and PyAnnote "\
        +"speaker turns on separate tiers, and is mostly useful for debugging and understanding "\
        +"the decisions that ASR and diarization are making independently."
    )
    parser.add_argument(
        "-n",
        "--num_speakers",
        help="Number of speakers in file. Default 2.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-m", "--asr_model",
        help=f"Whisper model. Larger models will ",
        default=ASR_CHOICES[1],
        choices=ASR_CHOICES,
    )
    parser.add_argument(
        "-d", "--drz_model",
        help=f"DRZ model path. Default is {DIARIZE_URI}.",
        default=DIARIZE_URI,
    )
    parser.add_argument(
        "-c",
        "--chunk_length_s",
        type=float,
        help="Chunk size to input to Whisper pipeline (default 30s). "\
        +"Note that this will not affect the size of chunks OUTPUT by Whisper, "\
        +"so this should only be modified if there is not enough memory to process "\
        +"30s at a time.",
        default=30,
    )
    parser.add_argument(
        "-w",
        "--return_word_timestamps",
        action='store_true',
        help="Whisper by default chunks speech more or less into utterances. "\
        +"Use this option to chunk by word, which may give more precise time accuracy "\
        +"When detecting speaker changes. (I recommend using this option.)"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If input is a directory, will search for files .wav files recursively"
    )
    parser.add_argument(
        '-b',
        "--asr_batch_size",
        type=int, default=8,
        help="Inference batch size for ASR. Default 8."
    )
    parser.add_argument(
        "-D", "--device",
        default=DEVICE,
        help=f"Device to run model on. Default {torch.device(DEVICE)}",
        choices=['cuda', 'cpu']
    )
    return parser

@Gooey
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return annotate(args)

if __name__ == '__main__':
    main()