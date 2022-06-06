from machine_learning import machine_learning
from reader import read_arff

ATTR_PATH = 'data/excitement.arff'
# ATTR_PATH = 'data/seriousness.arff'
# ATTR_PATH = 'data/comfortable.arff'
# ATTR_PATH = 'data/cheerfulness.arff'
SPEAKER_COUNT = 3

def main():
    dataset = read_arff(path = ATTR_PATH)
    machine_learning(dataset = dataset, speaker_count = SPEAKER_COUNT)

if __name__ == '__main__':
    main()
