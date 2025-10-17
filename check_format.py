from anticipation.vocab import *

# Read first test sequence
with open('data/test_output.txt') as f:
    tokens = list(map(int, f.readline().split()))

print(f'Total tokens: {len(tokens)}')
print(f'\nFirst 4 tokens (should be [ANTICIPATE, SEP, SEP, SEP]):')
print(f'  {tokens[:4]}')
print(f'  Correct: {tokens[:4] == [ANTICIPATE, SEPARATOR, SEPARATOR, SEPARATOR]}')

# Check prefix format
data = tokens[4:]
print(f'\nData tokens after first 4: {len(data)}')

print('\nChecking prefix (first 33 control/rest pairs):')
prefix_controls = 33
errors = 0

for i in range(prefix_controls):
    if i*6 + 6 > len(data):
        print(f'  {i}: Not enough tokens left')
        break
    
    ctrl = data[i*6:i*6+3]
    rest = data[i*6+3:i*6+6]
    
    ctrl_valid = ctrl[0] >= CONTROL_OFFSET
    rest_valid = rest[1] == DUR_OFFSET and rest[2] == REST
    
    if not ctrl_valid or not rest_valid:
        if errors < 5:  # Only print first 5 errors
            print(f'  {i}: Control={ctrl} (valid={ctrl_valid}), Rest={rest} (valid={rest_valid})')
        errors += 1

if errors == 0:
    print(f'  All {prefix_controls} prefix blocks are correctly formatted!')
else:
    print(f'  Found {errors} format issues in prefix')

# Check alternating pattern after prefix
print(f'\nChecking alternating pattern after prefix:')
start_pos = prefix_controls * 6
remaining = data[start_pos:]
print(f'  Remaining tokens: {len(remaining)}')
print(f'  First 30 tokens: {remaining[:30]}')

# Look for score/control alternation
alt_errors = 0
for i in range(min(10, len(remaining) // 6)):
    pos = i * 6
    if pos + 6 > len(remaining):
        break
    
    score = remaining[pos:pos+3]
    ctrl = remaining[pos+3:pos+6]
    
    score_valid = score[0] < CONTROL_OFFSET
    ctrl_valid = ctrl[0] >= CONTROL_OFFSET
    
    if not score_valid or not ctrl_valid:
        if alt_errors < 5:
            print(f'  {i}: Score={score} (not_control={score_valid}), Control={ctrl} (is_control={ctrl_valid})')
        alt_errors += 1

if alt_errors == 0:
    print(f'  All checked alternations are correct!')
else:
    print(f'  Found {alt_errors} alternation issues')
