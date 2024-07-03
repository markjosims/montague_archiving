from pympi import Elan
from typing import Optional, Sequence, Dict
from argparse import ArgumentParser
from glob import glob
import os

def write_script(eaf_fp: str) -> str:
    eaf = Elan.Eaf(eaf_fp)
    turns = []
    speakers = eaf.get_tier_names()
    for speaker in speakers:
        annotations = eaf.get_annotation_data_for_tier(speaker)
        for start, end, val in annotations:
            turns.append({'start': start, 'end': end, 'text': val, 'speaker': speaker})
    turns.sort(key=lambda d:d['start'])
    merged_turns = []
    i=0
    while i < len(turns)-1:
        turn = turns[i]
        next_turn = turns[i+1]
        if turn['speaker'] == next_turn['speaker']:
            merged_turn = merge_turns(turn, next_turn)

def merge_turns(turn1: Dict[str, str], turn2: Dict[str, str]) -> Dict[str, str]:
    merged_turn = {}
    merged_turn['start'] = turn1['start']
    merged_turn['end'] = turn2['end']
    merged_turn['text'] = ' '.join(turn1['text'], turn2['text'])

    return merged_turn

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser(description="Render .eaf files as text transcripts.")
    parser.add_argument("--input", "-i", help=".eaf file or folder of .eaf files")
    parser.add_argument("--recursive", '-r', help='If input is dir, find .eaf files recursively', action='store_true')
    args = parser.parse_args(argv)
    input_fp = args.input
    if os.path.isdir(input_fp):
        if args.recursive:
            glob_str = os.path.join(input_fp, '**', '*.eaf')
        else:
            glob_str = os.path.join(input_fp, '*.eaf')
        eafs = glob(glob_str, recursive=args.recursive)
        for eaf_fp in eafs:
            write_script(eaf_fp)
        return 0
    write_script(input_fp)