from argparse import ArgumentParser
import json
from string import punctuation
from eaf_to_script import ms_to_human_time
brackets = r"{[()]}"
non_bracket_punct = [p for p in punctuation if p not in brackets]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    with open(args.input, encoding='utf8') as f:
        gecko_obj = json.load(f)
    outpath = args.output or args.input.replace('.json', '.txt')
    with open(outpath, encoding='utf8', mode='w') as f:
        for turn in gecko_obj['monologues']:
            start_ms = int(turn['start']*1_000)
            start = ms_to_human_time(start_ms)
            end_ms = int(turn['end']*1_000)
            end = ms_to_human_time(end_ms)

            f.write(
                f"{turn['speaker']['id']} {start}-{end}: "
            )
            words = " ".join(term['text'] for term in turn['terms'])
            for p in non_bracket_punct:
                words = words.replace(" "+p, p)
            f.write(words)
            f.write('\n\n\n')
