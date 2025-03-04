import scipy.interpolate

def load_annotation_file(file_path):
    annotations = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                timestamp = float(parts[0])
                annotation = parts[2]
                annotations.append((timestamp, annotation))
    
    return annotations

def compare_annotations(file1_path, file2_path):
    """
    Creates a mapping between downbeat and beat times in two annotation files.
    Inputs are timestamps in the first file, outputs are timestamps in the second file
    """

    annotations1 = load_annotation_file(file1_path)
    annotations2 = load_annotation_file(file2_path)

    min_length = min(len(annotations1), len(annotations2))
    if len(annotations1) != len(annotations2):
        shorter_file = file1_path if len(annotations1) == min_length else file2_path
        print(f'Number of annotations in {file1_path} and {file2_path} do not match.')
        print(f"Proceeding with the first {min_length} annotations from {shorter_file}.")

    
    data = []

    for i in range(min_length):   
        data.append((annotations1[i][0], annotations2[i][0]))

    x,y = list(zip(*data))
    map = scipy.interpolate.interp1d(x, y)
    
    return map