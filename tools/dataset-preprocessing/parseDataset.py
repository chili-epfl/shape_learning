import sys
import re
import codecs

letter_re = re.compile("WORD (?P<char>.)")
strokes_re = re.compile("NUMSTROKES (?P<nb>\d)")
points_re = re.compile("POINTS (?P<nb>\d*) # (?P<coords>[-\d\s]*)")


def parse(dataset):
    shapes = {}

    with codecs.open(dataset,'r', encoding="utf8") as dataset:
        current_char = ''
        remaining_strokes = 0
        current_shape = []
        for l in dataset.readlines():
            letter = letter_re.search(l)
            if letter:

                current_char = letter.group('char')
                continue

            strokes = strokes_re.search(l)
            if strokes:
                # if we already parsed one letter, add it to the shape dictionary
                if current_shape:
                    shapes.setdefault(current_char,[]).append(current_shape)
                    current_shape = []

                remaining_strokes = int(strokes.group('nb'))
                continue

            points = points_re.search(l)
            if points:
                if remaining_strokes == 0:
                    raise RuntimeError("I should not find points! No stroke is missing")
                remaining_strokes -= 1
                current_shape.append(map(int,points.group("coords").split()))

    return shapes

def write_samples(dataset, path):
    import os.path

    for char, samples in dataset.items():
        if char.startswith("\\"):
            # skip
            continue

        with open(os.path.join(path,"%s.dat" % char), "w") as letter:
                letter.write("%s\n" % len(samples))
                letter.write("%s\n" % (len(samples[0])/2))
                for sample in samples:
                    letter.write(" ".join(map(str,sample)) + "\n")


if __name__ == "__main__":

    from preprocessDataset import preprocess

    if len(sys.argv) < 2:
        print("Usage: parseDataset.py <dataset> [path]")
        sys.exit(1)

    if len(sys.argv) == 3:
        path = sys.argv[2]
    else:
        path = ""

    shapes = parse(sys.argv[1])
    print("Found %d shapes" % len(shapes))
    samples = preprocess(shapes)
    for k,v in samples.items():
        print("%s: %s samples" % (k, len(v)))
    print("Writing them to %s..." % path)
    write_samples(samples, path)

