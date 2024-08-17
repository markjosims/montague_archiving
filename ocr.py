from doctr.models import detection_predictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import extract_crops
from doctr.io import DocumentFile
import numpy as np
from PIL import Image
from transformers import pipeline, Pipeline
from argparse import ArgumentParser
from typing import Optional, Sequence
import os
from glob import glob

OCR_HAND = 'microsoft/trocr-large-handwritten'
OCR_PRINT = 'microsoft/trocr-large-printed'

def get_line_boxes(boxes, lines):
    line_boxes = np.zeros([len(lines), 4])
    for i, line in enumerate(lines):
        these_boxes = [boxes[j] for j in line]
        
        line_xmins = [box[0] for box in these_boxes]
        line_xmin = min(line_xmins)
        line_boxes[i][0]=line_xmin

        line_ymins = [box[1] for box in these_boxes]
        line_ymin = min(line_ymins)   
        line_boxes[i][1]=line_ymin

        line_xmaxs = [box[2] for box in these_boxes]
        line_xmax = max(line_xmaxs)
        line_boxes[i][2]=line_xmax

        line_ymaxs = [box[3] for box in these_boxes]
        line_ymax = max(line_ymaxs)
        line_boxes[i][3]=line_ymax
    return line_boxes

def perform_ocr(img_fp: str, text_detector: DetectionPredictor, ocr_pipe: Pipeline ) -> str:
    img_doc = DocumentFile.from_images(img_fp)
    out_txt = ''

    text_preds = text_detector(img_doc)
    for i, page_text in enumerate(text_preds):
        page = img_doc[i]
        out_txt+=f'PAGE {i+1}\n\n'
        word_boxes = page_text['words']
        line_groups = DocumentBuilder()._resolve_lines(word_boxes)
        line_boxes = get_line_boxes(word_boxes, line_groups)
        crops = extract_crops(page, line_boxes)
        crop_imgs = [Image.fromarray(crop) for crop in crops]
        result = ocr_pipe(crop_imgs)
        for line in result:
            line_txt = ' '.join(chunk['generated_text'] for chunk in line)
            out_txt+=line_txt+'\n'
    return out_txt

def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Image to run OCR on or folder of images.')
    parser.add_argument('-o', '--output', help='Folder to output to. Defaults to input folder.')
    parser.add_argument('-d', '--doc_type', choices=['hand', 'print'], help='Document type, handwritten or printed.')
    # parser.add_argument('--perform_correction', '-c', help='Whether to run correction on OCR output with OCRonos. Can be time intensive.')
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = init_parser()
    args = parser.parse_args(argv)
    return ocr(args)

def ocr(args) -> int:
    print('Loading text detector...')
    text_detector = detection_predictor(pretrained=True)
    print('Loading OCR model...')

    ocr_model = OCR_PRINT if args.doc_type == 'print' else OCR_HAND
    ocr_pipe = pipeline("image-to-text", model=ocr_model, max_new_tokens=128)
    
    input_path = args.input
    output_path = args.output or input_path

    jpgs = [input_path,]

    if os.path.isdir(input_path):
        if not os.path.isdir(output_path):
            print("Input argument is folder but output is file. Setting output to parent directory of file.")
            output_path = os.path.dirname(output_path)
        jpgs = glob(os.path.join(input_path,'*.jpg'))

    for jpg in jpgs:
        print(f'Performing OCR on {jpg}...')
        result = perform_ocr(jpg, text_detector, ocr_pipe)
        jpg_basename = os.path.basename(jpg)
        result_path = os.path.join(output_path, jpg_basename+'.txt')
        print(f"Writing output for image {jpg} to {result_path}")
        with open(result_path, 'w', encoding='utf8') as f:
            f.write(result)

    return 0

if __name__ == '__main__':
    main()