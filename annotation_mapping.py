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
    keys are timestamps in the first file, values are timestamps in the second file
    """

    annotations1 = load_annotation_file(file1_path)
    annotations2 = load_annotation_file(file2_path)

    if len(annotations1) != len(annotations2):
        raise ValueError("Number of annotations do not match")
    
    mapping = {}

    for i, item in enumerate(annotations1):   
        mapping[item[0]] = annotations2[i][0]
    
    return mapping