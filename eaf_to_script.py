from pympi import Elan
from typing import Optional, Sequence, Dict, List
from argparse import ArgumentParser
from glob import glob
import os

def human_time(timestamp: float) -> str:
    hours = timestamp//3600
    minutes = timestamp%3600//60
    seconds = int(timestamp%60)
    return f"{hours}:{minutes:0>2d}:{seconds:0>2d}"

def write_script(eaf_fp: str, out_fp: str) -> str:
    eaf = Elan.Eaf(eaf_fp)
    turns = []
    speakers = eaf.get_tier_names()
    for speaker in speakers:
        annotations = eaf.get_annotation_data_for_tier(speaker)
        for start, end, val in annotations:
            turns.append({'start': start, 'end': end, 'text': val, 'speaker': speaker})
    merged_turns = merge_turn_list(turns)

    with open(out_fp, 'w') as f:
        for turn in merged_turns:
            speaker = turn['speaker']
            start = human_time(turn['start'])
            end = human_time(turn['end'])
            f.write(f"{speaker}: {start}-{end}")
            f.write(turn['text'])
            f.write("\n")

def merge_turn_list(turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
    turns = sorted(turns, key=lambda d:d['start'])
    merged_turns = []
    i=0
    while i < len(turns):
        turn = turns[i]
        next_turn = turns[i+1] if i<len(turns)-1 else None
        i+=1
        while next_turn and (turn['speaker'] == next_turn['speaker']):
            turn = merge_turn_pair(turn, next_turn)
            next_turn = turns[i+1] if i<len(turns)-1 else None
            i+=1
        merged_turns.append(turn)
    
    return merged_turns

def merge_turn_pair(turn1: Dict[str, str], turn2: Dict[str, str]) -> Dict[str, str]:
    merged_turn = {}
    merged_turn['start'] = turn1['start']
    merged_turn['end'] = turn2['end']
    merged_turn['text'] = ' '.join([turn1['text'], turn2['text']])
    merged_turn['speaker'] = turn1['speaker']

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
            out_fp = eaf_fp.replace('.eaf', '.txt')
            write_script(eaf_fp, out_fp)
        return 0
    out_fp = input_fp.replace('.eaf', '.txt')
    write_script(input_fp, out_fp)
    return 0

if __name__ == '__main__':
    main()