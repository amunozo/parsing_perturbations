from attacks import get_random_attack
import os
import random
import tempfile

ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'


def perturbate_file(treebank, perturbation, file=None):
    """
    Perturbate a test file applying a random attack to the inner letters of the percentage of content words selected.
    """
    # Locate the test file
    if not file:
        for filename in os.listdir(ud_folder + treebank):
            if filename.endswith('test.conllu'):
                test_conllu = ud_folder + treebank + '/' + filename

        # Read test file
        with open(test_conllu, 'r') as f:
            test = f.readlines()
    else:
        with open(file, 'r') as f:
            test = f.readlines()

    # Apply the perturbation with a "perturbation" probability only to the content words
    perturbed_test = []
    content_words = {'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'}

    for line in test:
        if line.startswith('#') or line == '\n':
            perturbed_test.append(line)

        else:
            line = line.split('\t')
            if line[3] in content_words:
                if 100*random.random() < perturbation:
                    line[1] = get_random_attack(line[1], treebank)

            perturbed_test.append('\t'.join(line))
    
    perturbed_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(perturbed_file.name, 'w') as f:
        f.write(''.join(perturbed_test))           

    return perturbed_file      
    
    
if __name__ == '__main__':
    for perturbation in [1]:
        file = perturbate_file('UD_Spanish-AnCora', perturbation)
        #with open(file.name, 'r') as f:
        #    print(f.read()[:1500])