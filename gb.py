import os
import tempfile
from perturbate import perturbate_file
import numpy as np
import tagger

ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'

def conllu_to_conll(input_file, output_file, test=False) -> None:
    print(f"INFO: Converting {input_file} file to {output_file} file")
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

def evaluate(treebank, tags, perturbation, perturbed_tagging, epochs=10, device=0):
    """
    Evaluate a GB parser on the perturbed test set of a treebank using a certain percentage of
    perturbed words and averaging the result over certain number of epochs.
    """
    # Locate test file
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

    model = model_folder + 'mod.model'

    for filename in os.listdir(ud_folder + treebank):
        if filename.endswith('test.conllu'):
            gold_conllu = ud_folder + treebank + '/' + filename
    
    # Perturbate test file
    uas_list = []
    las_list = []
    acc_list = []

    print("Treebank: ", treebank)
    print("Tags: ", tags)
    print("Perturbation: ", perturbation)
    print("Perturbed tagging: ", perturbed_tagging)

    if perturbation != 0:
        for i in range(epochs):
            print("Epoch: ", i+1)
            if tags != 'none':
                # Predicts PoS tags; it should give a list with lists of predicted tags to insert in the conllx file
                if perturbed_tagging == 'perturbed':
                    perturbed_test_conllu = perturbate_file(treebank, perturbation)
                    predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, perturbed_test_conllu.name)
                else:
                    predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, gold_conllu)
                    predicted_test_conllu = perturbate_file(treebank, perturbation, predicted_test_conllu.name)
                
                acc = tagger.evaluate(treebank, tags, gold_conllu, test_seq_pred)
                
                # Measure tagger accuracy
                acc_list.append(acc)
            
                perturbed_test_conllx = conllu_to_conll(predicted_test_conllu.name, None, test=True)
            else:
                perturbed_test_conllu = perturbate_file(treebank, perturbation)
                perturbed_test_conllx = conllu_to_conll(perturbed_test_conllu.name, None, test=True)

            # Get UAS and LAS
            evaluate_str = 'python -u -m supar.cmds.biaffine_dep evaluate \
            --path {} \
            --data {} \
            --device {} \
            --punct --tree'.format(model, perturbed_test_conllx, device)
            output = os.popen(evaluate_str).read()
            score = output.split('\n')[-3].split(' ')
            if score[-3] != 'LAS:':
                UAS = score[-3].replace('%', '')
            else:
                UAS = score[-4].replace('%', '')

            LAS = score[-1].replace('%', '')
            uas_list.append(float(UAS))
            las_list.append(float(LAS))
            print('UAS: ', UAS, 'LAS: ', LAS)
    
    else:
        epochs = 1
        if tags != 'none':
            predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, gold_conllu)
            acc = tagger.evaluate(treebank, tags, gold_conllu, test_seq_pred) #, test_conllu)
            acc_list.append(acc)

        else:
            acc = 100
            acc_list.append(acc)
            predicted_test_conllu = gold_conllu

        # Get UAS and LAS
        evaluate_str = 'python -u -m supar.cmds.biaffine_dep evaluate \
        --path "{}" \
        --data "{}" \
        --device "{}" \
        --punct --tree'     
        try:
            perturbed_test_conllx = conllu_to_conll(predicted_test_conllu, None, test=True)
            
        except:
            perturbed_test_conllx = conllu_to_conll(predicted_test_conllu.name, None, test=True)
        
        evaluate_str = evaluate_str.format(model, perturbed_test_conllx, device)

        output = os.popen(evaluate_str).read()
        score = output.split('\n')[-3].split(' ')
        if score[-3] != 'LAS:':
            UAS = score[-3].replace('%', '')
        else:
            UAS = score[-4].replace('%', '')
        
        LAS = score[-1].replace('%', '')
        uas_list.append(float(UAS))
        las_list.append(float(LAS))
        print('UAS: ', UAS, 'LAS: ', LAS)
    
    average_uas = sum(uas_list) / len(uas_list)
    average_las = sum(las_list) / len(las_list)
    std_uas = np.std(uas_list)
    std_las = np.std(las_list)
    if tags != 'none':    
        average_acc = sum(acc_list) / len(acc_list)
        std_acc = np.std(acc_list)
    else:
        average_acc = np.nan
        std_acc = 0

    print("Average UAS: ", round(average_uas,2), "Standard deviation: ", round(std_uas,2))
    print("Average LAS: ", round(average_las,2), "Standard deviation: ", round(std_las,2))
    print('')

    # Create the .csv file if it doesn't exist
    csv_file = 'scores.csv'

    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('Treebank,Parser,Tags,Perturbed tagging,Perturbation,UAS,UAS std,LAS,LAS std,Tagger acc,Tagger acc std,Epochs\n')
    # Write the results in the .csv file
    with open(csv_file, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            treebank, 'GB', tags, perturbed_tagging, perturbation, average_uas, std_uas, average_las, std_las, average_acc, std_acc, epochs
        )
    )


if __name__ == '__main__':
    evaluate('UD_Spanish-AnCora', 'upos', 0, 'perturbed', epochs=10)