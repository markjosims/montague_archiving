import pyrubberband
import soundfile as sf
from argparse import ArgumentParser
    
def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-r', '--rate', type=float)
    
    args = parser.parse_args()

    y, sr = sf.read(args.input)
    
    y_stretched = pyrubberband.time_stretch(y, sr, rate=args.rate) 
    
    sf.write(args.output, y_stretched, sr)

if __name__ == '__main__':
    main()