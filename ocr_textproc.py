from argparse import ArgumentParser
from typing import *
from glob import glob
import os

"""
Make all text lowercase, except at beginning of sentence.
"""

def main(argv: Optional[Sequence[str]]=None):
    parser = ArgumentParser()
    parser.add_argument("--input", '-i')
    parser.add_argument("--output", '-o')

    args = parser.parse_args(argv)
    for textfile in glob(os.path.join(args.input, '*.txt')):
        ...

if __name__ == '__main__':
    main()