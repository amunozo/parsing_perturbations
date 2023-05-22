import os

ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'

def prepare_data(treebank, tags, model_folder):
    """
    Encode a CoNLLu file in a format usable by the NCRF++ given a type of tags and a UD treebank
    """
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    # Select train and dev file
    files = []
    for filename in os.listdir(ud_folder + treebank):
        if filename.endswith('ud-train.conllu'):
            train_conllu = ud_folder + treebank + '/' + filename
            train = filename
            print(filename)
            files.append((train_conllu, train))
        elif filename.endswith('ud-dev.conllu'):
            dev_conllu = ud_folder + treebank + '/' + filename
            dev = filename
            files.append((dev_conllu, dev))
        elif filename.endswith('ud-test.conllu'):
            test_conllu = ud_folder + treebank + '/' + filename
            test = filename
            files.append((test_conllu, test))

    if tags == 'none':
        tags = 'upos'

    # Encode the files
    encoding_script = 'dep2label/encode_dep2labels.py'
    encoded_folder = model_folder + 'seq_files/'
    if not os.path.exists(encoded_folder):
        os.makedirs(encoded_folder)
    
    for file in files:
        encoding_command = 'python "{}" --input "{}" --output "{}"  --encoding "2-planar-brackets-greedy" --mtl "3-task" --tag "{}"'.format(
            encoding_script, file[0], encoded_folder + file[1].replace('conllu', 'seq'), tags
        )
        os.system(encoding_command)


def conllu_to_conll(input_file, output_file, test=False) -> None:
    #print(f"INFO: Converting {input_file} file to {output_file} file")
    if not test:
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
    
    else:
        output_file = 'temp/output'
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