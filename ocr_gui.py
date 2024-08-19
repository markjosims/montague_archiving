from ocr import ocr
from typing import Optional, Sequence
from gooey import Gooey, GooeyParser

def init_parser() -> GooeyParser:
    parser = GooeyParser()
    parser.add_argument('-i', '--input', help='Image to run OCR on or folder of images.', widget='DirChooser')
    # parser.add_argument('-o', '--output', help='Folder to output to. Defaults to input folder.', widget='DirChooser')
    parser.add_argument('-d', '--doc_type', choices=['hand', 'print'], help='Document type, handwritten or printed.')
    # parser.add_argument('--perform_correction', '-c', help='Whether to run correction on OCR output with OCRonos. Can be time intensive.')
    return parser

@Gooey
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    args.output = args.input
    return ocr(args)

if __name__ == '__main__':
    main()