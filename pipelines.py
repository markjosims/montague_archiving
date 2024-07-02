from typing import Optional, Sequence, Dict, List, Union
from argparse import ArgumentParser
from transformers import Pipeline, pipeline
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import torchaudio
import json
import numpy as np
from pympi import Elan

SAMPLE_RATE = 16000

"""
Pyannote and HuggingFace entry points
"""

def perform_asr() -> None:
    ...

def perform_vad(
        in_fp: str,
        pipe: Optional[PyannotePipeline] = None,
    ):
    wav = load_and_resample(in_fp)

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/voice-activity-detection")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": SAMPLE_RATE},
            hook=hook,
        )
    return result

def pyannote_result_to_json(result)  -> List[Dict[str, float]]:
    segments = []
    for track, _ in result.itertracks():
        segment = {'start': track.start, 'end': track.end}
        segments.append(segment)
    return segments

def diarize(
        in_fp: str,
        pipe: Optional[PyannotePipeline] = None,
        num_speakers: int = 1,
    ):
    wav = load_and_resample(in_fp)

    if not pipe:
        pipe = PyannotePipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    with ProgressHook() as hook:
        result = pipe(
            {"waveform": wav, "sample_rate": SAMPLE_RATE},
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

def load_and_resample(fp: str) -> torch.Tensor:
    wav_orig, sr_orig = torchaudio.load(fp)
    wav = torchaudio.functional.resample(wav_orig, sr_orig, SAMPLE_RATE)
    return wav

def sec_to_samples(time_sec: float):
    """`time_sec` is a time value in seconds.
    Returns same time value in samples using
    global constant SAMPLE_RATE.
    """
    return int(time_sec*SAMPLE_RATE)

def segments_to_array(
        segments: List[Dict[str, float]],
        len_samples: Optional[int] = None
    ) -> np.ndarray:
    """
    `segments` is a list of dicts containing start and end timestamps.
    Return a numpy array of samples (scaled to the SAMPLING_RATE global
    constant) containing 1s for samples corresponding to segment timestamps
    and 0s elsewhere. Array will have length of `len_samples` if passed,
    else the sample length corresponding to the maximum timestamp in
    `segments`.
    """
    if not len_samples:
        max_time = segments[-1]['end']
        len_samples = sec_to_samples(max_time)

    array = np.zeros(len_samples)
    for seg in segments:
        start_sample = sec_to_samples(seg['start'])
        end_sample = sec_to_samples(seg['end'])
        array[start_sample:end_sample] = 1
    
    return array

def remove_segments_from_audio(
        audio: Union[torch.Tensor, np.ndarray],
        ipa_segments: List[Dict[str, float]],
        sli_segments: List[Dict[str, float]],
    ) -> torch.tensor:
    """
        `audio` is a torch tensor or numpy array containing audio samples.
        `ipa_segments` is a list of dicts containing start and end times
        for ipa annotations from an Elan file,
        and `sli_segments` is a list of speech segments classified as Tira
        by an SLI model.
        This function masks all samples which correspond to `sli_segments`
        but not `ipa_segments`, then cuts them from the audio.
        Returns a torch tensor containing all audio samples except for those
        cut.
    """
    if len(audio.shape) == 2:
        audio = audio[0]
    if type(audio) is torch.Tensor:
        audio = audio.numpy()

    vad_mask = segments_to_array(sli_segments, len_samples=len(audio))
    ipa_mask = segments_to_array(ipa_segments, len_samples=len(audio))
    ipa_mask*=2
    mask = vad_mask + ipa_mask
    # 0 - no annotation
    # 1 - machine label annotation only
    # 2 - ipa annotation only
    # 3 - machine label + ipa annotation
    audio = audio[mask!=1]

    # reshape to 2D torch tensor
    audio = torch.unsqueeze(torch.from_numpy(audio), 0)

    return audio

"""
Main script
"""

def init_parser() -> ArgumentParser:
    parser = ArgumentParser("VAD, diarization and ASR runner")
    parser.add_argument(
        "TASK",
        help="Task to run",
        choices=["VAD", "ASR", "DRZ"]
    )
    parser.add_argument("-i", "--input", help="Filepath to run VAD on")
    parser.add_argument("-o", "--output", help="Filepath to save predictions to")
    parser.add_argument(
        "-n",
        "--num_speakers",
        help="Number of speakers in file (only use for diarization)",
        default=1
    )
    parser.add_argument("-m", "--model", help="Model path (if overriding default).")
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)

    if args.TASK == 'ASR':
        labels = perform_asr(args.input)
    elif args.TASK == 'VAD':
        result = perform_vad(args.input)
        labels = pyannote_result_to_json(result)
    elif args.TASK == 'DRZ':
        result = diarize(args.input, num_speakers=args.num_speakers)
        labels = pyannote_result_to_json(result)

    out_fp = args.output or args.input.replace('.wav', '.json')
    with open(out_fp, 'w') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return 0



if __name__ == '__main__':
    main()