from eaf_to_script import human_time_to_ms, ms_to_human_time
import re

if __name__ == '__main__':
    fp = r'annotations\15-03-17_Robert-Wall.txt'
    with open(fp) as f:
        lines = f.readlines()

    i = 681
    assert lines[i] == 'RW: 692:16:55-694:14:12\n'

    bad_timestamps = lines[:i+1]
    good_timestamps = lines[i+1:]

    is_timestamp=re.compile(r'\d+:\d+:\d+')
    fixed_timestamps = []
    for line in bad_timestamps:
        matches = is_timestamp.findall(line)
        for match in matches:
            bad_ms = human_time_to_ms(match)
            real_ms = bad_ms//1000
            fixed_timestamp = ms_to_human_time(real_ms)
            line = line.replace(match, fixed_timestamp)
        fixed_timestamps.append(line)
    
    out_lines = fixed_timestamps + good_timestamps
    out_fp = fp.replace('.txt', '-fixed-timestamps.txt')
    with open(out_fp, 'w') as f:
        f.writelines(out_lines)
            