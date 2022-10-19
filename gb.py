import os
from perturbate import perturbate_file


ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'

def conllu_to_conll(input_file, output_file) -> None:
    print(f"INFO: Converting {input_file} file to {output_file} file")
    with open(input_file, 'rt', encoding='UTF-8', errors="replace") as conllu, open(output_file, 'wt', encoding='UTF-8',
                                                                                    errors="replace") as conll:
        for line in conllu:
            if line != "\n":
                tuples = line.split("\t")
                if len(tuples) == 10 and tuples[0] != '#' and '.' not in tuples[0] and '-' not in tuples[0]:
                    tuples[8] = tuples[9] = '_'
                    conll.write('\t'.join(tuples) + '\n')
            else:
                conll.write('\n')
    return output_file

def train(treebank, tags, device):
    """
    Train a SuPar dependency parser using a UD treebank
    """
    if tags == 'none':
        model_folder = 'supar_models/none_models/' + treebank + '/'
    
    else:
        model_folder = 'supar_models/tags_models/'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'
        
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model = model_folder + 'mod.model'

    # Select train and dev file
    for filename in os.listdir(ud_folder + treebank):
        if filename.endswith('train.conllu'):
            train_conllu = ud_folder + treebank + '/' + filename
            train_conllx = conllu_to_conll(train_conllu, model_folder + 'train.conllx')
        elif filename.endswith('dev.conllu'):
            dev_conllu = ud_folder + treebank + '/' + filename
            dev_conllx = conllu_to_conll(dev_conllu, model_folder + 'dev.conllx')
        
    # Find word embeddings
    language = treebank.split('-')[0].replace('UD_', '').lower()
    embeddings = 'embeddings/' + language

    if tags == 'upos':
        tags = 'tag'
    elif tags == 'none':
        tags = ''

    # Train the model
    train_command = 'python -u -m supar.cmds.biaffine_dep train -b -d 0 -c biaffine-dep-en -p model -f char {}\
    --train "{}"  \
    --dev "{}"  \
    --test "{}"  \
    --path "{}" \
    --device "{}" \
    --embed "{}" \
    --punct --tree \
    --n-embed 300 \
    --unk='.format(tags, train_conllx, dev_conllx, dev_conllx, model, device, embeddings)

    os.system(train_command)

def evaluate(treebank, tags, perturbation, epochs=10, device=0):
    """
    Evaluate a GB parser on the perturbed test set of a treebank using a certain percentage of
    perturbed words and averaging the result over certain number of epochs.
    """

if __name__ == '__main__':
    train('UD_Irish-IDT', 'none', '1')