from typing import Optional, Sequence, Dict, List, Union
from argparse import ArgumentParser
from transformers import Pipeline, pipeline
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from eaf_to_script import write_script
import torch
import torchaudio
import numpy as np
from pympi import Elan
from glob import glob
import os
from tqdm import tqdm

SAMPLE_RATE = 16000
DIARIZE_URI = "pyannote/speaker-diarization-3.1"
ASR_URI = "openai/whisper-medium.en"
DEVICE = 0 if torch.cuda.is_available() else -1

"""
Pyannote and HuggingFace entry points
"""

def perform_asr(
        audio: Union[torch.Tensor, np.ndarray],
        pipe: Optional[Pipeline] = None,
    ) -> str:
    if not pipe:
        pipe = pipeline("automatic-speech-recognition", model=ASR_URI)
    if type(audio) is torch.Tensor:
        audio = np.array(audio[0,:])
    result = pipe(audio)
    return result["text"]

def diarize(
        audio: torch.Tensor,
        pipe: Optional[PyannotePipeline] = None,
        num_speakers: int = 2,
    ):

    if not pipe:
        pipe = PyannotePipeline.from_pretrained(DIARIZE_URI)

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": audio, "sample_rate": SAMPLE_RATE},
            num_speakers=num_speakers,
            hook=hook,
        )
    return result

"""
ELAN methods
"""

def get_ipa_labels(elan_fp: str) -> List[Dict[str, Union[str, float]]]:
    """
    Read data from IPA Transcription tier in a .eaf file
    indicated by `elan_fp`. Return list of dicts containing
    start time, end time and value for each annotation.
    """
    eaf = Elan.Eaf(elan_fp)
    ipa_tuples = eaf.get_annotation_data_for_tier('IPA Transcription')
    ipa_labels = [{'start': a[0], 'end': a[1], 'value': a[2]} for a in ipa_tuples]
    return ipa_labels
    

"""
Audio handling methods
"""

def load_and_resample(fp: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, sr)
    return wav

def sec_to_samples(time_sec: float) -> int:
    """`time_sec` is a time value in seconds.
    Returns same time value in samples using
    global constant SAMPLE_RATE.
    """
    return int(time_sec*SAMPLE_RATE)

def sec_to_ms(time_sec: float) -> int:
    return int(time_sec*1000)

def get_segment_slice(
        audio: torch.Tensor,
        segment,
    ) -> np.ndarray:
    """
    Takes torchaudio tensor and a pyannote segment,
    returns slice of tensor corresponding to segment endpoints.
    """
    start_idx = sec_to_samples(segment.start)
    end_idx = sec_to_samples(segment.end)
    return audio[:,start_idx:end_idx]

"""
Main script
"""

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("Annotation runner")
    parser.add_argument("-i", "--input", help="Directory of files to annotate")
    parser.add_argument("-s", "--strategy", choices=["asr-only", "asr-first", "drz-first"])
    parser.add_argument(
        "-n",
        "--num_speakers",
        help="Number of speakers in file",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-m", "--asr_model",
        help="ASR model path (if overriding default).",
        default=ASR_URI,
    )
    parser.add_argument(
        "-d", "--drz_model",
        help="DRZ model path (if overriding default).",
        default=DIARIZE_URI,
    )
    parser.add_argument(
        "-D", "--device",
        default=DEVICE,
        type=int,
    )
    parser.add_argument(
        "-c", "--chunk_len_s", type=float, help="Chunk size to use for ASR pipeline."
    )
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    print(f"Initializing ASR pipeline from URI {args.asr_model}...")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=args.device,
        chunk_length_s=args.chunk_len,
    )
    if args.strategy != "asr-only":
        print(f"Initializing diarization pipeline from URI {args.drz_model}...")
        drz_pipe = PyannotePipeline.from_pretrained(args.drz_model)
        drz_pipe.to(args.device)

    wav_fps = glob(os.path.join(args.input, "*.wav"))
    for wav_fp in wav_fps:
        print("Annotating file", wav_fp)
        eaf = Elan.Eaf()
        eaf.add_linked_file(wav_fp)
        wav = load_and_resample(wav_fp)
        if args.strategy=='drz_first':
            eaf = drz_first(
                wav=wav,
                eaf=eaf,
                num_speakers=args.num_speakers,
                drz_pipe=drz_pipe,
                asr_pipe=asr_pipe,
            )
        elif args.strategy=='asr_first':
            eaf = asr_first(
                wav=wav,
                eaf=eaf,
                num_speakers=args.num_speakers,
                drz_pipe=drz_pipe,
                asr_pipe=asr_pipe,
            )
        else:
            eaf = asr_only(
                wav=wav,
                eaf=eaf,
                asr_pipe=asr_pipe
            )

        eaf_fp = wav_fp.replace('.wav', '.eaf')
        eaf.to_file(eaf_fp)
        txt_fp = wav_fp.replace('.wav', '.txt')
        write_script(eaf, txt_fp)

        print("Saved ELAN annotations to", eaf_fp)
        print("Saved text annotations to", txt_fp)

    return 0

def asr_only(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        asr_pipe: Pipeline,
    ):
    chunks = asr_pipe(wav, batch_size=8, return_timestamps=True)["chunks"]
    for chunk in chunks:
        start, end = chunk['timestamp']
        text = chunk['text']
        eaf.add_annotation("default", start, end, text)
    return eaf

def asr_first(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
        asr_pipe: Pipeline,
    ):
    raise NotImplementedError("Not yet bucko.")

def drz_first(
        wav: torch.Tensor,
        eaf: Elan.Eaf,
        num_speakers: int,
        drz_pipe: PyannotePipeline,
        asr_pipe: Pipeline,
    ):

    diarization = diarize(wav, drz_pipe, num_speakers=num_speakers)

    speakers = diarization.labels()
    for speaker in speakers:
        eaf.add_tier(speaker)
        speaker_timeline = diarization.label_timeline(speaker)
        for segment in tqdm(
                speaker_timeline,
                desc=f"Performing ASR for speaker {speaker}",
                total=len(list(speaker_timeline
            ))):
            segment_wav = get_segment_slice(wav, segment)
            segment_text = perform_asr(segment_wav, asr_pipe)
            start_ms = sec_to_ms(segment.start)
            end_ms = sec_to_ms(segment.end)
            eaf.add_annotation(speaker, start_ms, end_ms, segment_text)
    
    return eaf


if __name__ == '__main__':
    main()