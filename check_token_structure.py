with open('data/test_sliding.txt') as f:
    line = f.readline().strip()
    tokens = [int(t) for t in line.split('|')[0].split()]

from anticipation.vocab import CONTROL_OFFSET, REST

start = 4
print('Triplets starting from index 4:')
for i in range(start, min(start+30, len(tokens)), 3):
    if i+2 < len(tokens):
        t, d, n = tokens[i], tokens[i+1], tokens[i+2]
        is_ctrl = 'CTRL' if t >= CONTROL_OFFSET else 'SCORE'
        print(f'  [{i}:{i+3}] {is_ctrl}: time={t}, dur={d}, note={n}')
