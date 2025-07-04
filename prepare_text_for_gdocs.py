from argparse import ArgumentParser
import re
from string import Template
import os

HEADER = Template("""
Filename:  $filename
Date of interview:  $interview_date
Date of transcription:  $transcription_date
Transcriber:  $transcriber
$participants

--- BEGIN TRANSCRIPTION ---
""")

FOOTER = "\n\n--- END TRANSCRIPTION ---"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input", '-i')
    parser.add_argument("--interview_date")
    parser.add_argument("--transcription_date")
    parser.add_argument("--transcriber", default="Mark Simmons")
    parser.add_argument("--participants", nargs='+')

    args = parser.parse_args()
    with open(args.input, encoding='utf8') as f:
        text = f.read()
    
    # preserve double newlines
    text = text.replace("\n\n", "<DOUBLELINEBREAK>")

    # condense newlines within turn
    newline_re = re.compile(r"(\n)^(\S)", re.MULTILINE)
    text = newline_re.sub(r" \2", text)

    # restore double newlines
    text = text.replace("<DOUBLELINEBREAK>", "\n\n")

    # trim leading whitespace
    whitespace_re = re.compile(r"^ ", re.MULTILINE)
    text = whitespace_re.sub('', text)

    # add newline before each speaker turn
    speaker_re = re.compile(r"^([A-Z][A-Z]:)", re.MULTILINE)
    text = speaker_re.sub(r"\n\n\1", text)

    # remove newline sequences in excess of 2
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # add header
    recording_name = os.path.splitext(os.path.basename(args.input))[0] + '.mp3'
    header_txt = HEADER.substitute(
        filename=recording_name,
        interview_date=args.interview_date,
        transcription_date=args.transcription_date,
        transcriber=args.transcriber,
        participants="\n".join(args.participants),
    )

    text = header_txt + text + FOOTER

    with open(args.input.replace('.txt', '-gdoc.text'), encoding='utf8', mode='w') as f:
        f.write(text)