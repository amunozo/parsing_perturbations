import os
from utils import prepare_data
from sklearn.metrics import accuracy_score
import tempfile
import numpy as np
import sys

ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'

def train(treebank, tags, device, learning_rate=0.02):
    """
    Train a tagger using NCRF++
    """
    if tags == 'none':
        model_folder = 'tagger_models/none_models/' + treebank + '/'
        features = 'False'
    
    else:
        model_folder = 'tagger_models/tags_models/'
        features = 'True'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'
        
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    prepare_data(treebank, tags, model_folder)
    
    encoded_folder = model_folder + 'seq_files/'
    for filename in os.listdir(encoded_folder):
        if filename.endswith('ud-train.seq'):
            train_seq = encoded_folder + filename

        elif filename.endswith('ud-dev.seq'):
            dev_seq = encoded_folder + filename
        
        elif filename.endswith('ud-test.seq'):
            test_seq = encoded_folder + filename
    
    # Remove the last column of the seq files
    with open(train_seq, 'r') as f:
        train = f.read().splitlines()
        # Remove third column
        train = [item.split('\t')[:2] for item in train]
        # rejoin the list
        train = ['\t'.join(item) for item in train]
        # rejoin as a file
        train = '\n'.join(train)
        train += '\n'
        with open(train_seq, 'w') as f:
            f.write(train)

    with open(dev_seq, 'r') as f:
        dev = f.read().splitlines()
        # Remove third column
        dev = [item.split('\t')[:2] for item in dev]
        # rejoin the list
        dev = ['\t'.join(item) for item in dev]
        # rejoin as a file
        dev = '\n'.join(dev)
        dev += '\n'
        with open(dev_seq, 'w') as f:
            f.write(dev)
    
    with open(test_seq, 'r') as f:
        test = f.read().splitlines()
        # Remove third column
        test = [item.split('\t')[:2] for item in test]
        # rejoin the list
        test = ['\t'.join(item) for item in test]
        # rejoin as a file
        test = '\n'.join(test)
        test += '\n'
        with open(test_seq, 'w') as f:
            f.write(test)

    for filename in os.listdir(ud_folder + treebank):
            if filename.endswith('dev.conllu'):
                gold_file = ud_folder + treebank + '/' + filename

                 
    # Find word embeddings
    language = treebank.split('-')[0].replace('UD_', '').lower()
    embeddings = 'embeddings/' + language

    # Create the config file
    # Now we localize where the main.py file is to train the model
    main_py = "dep2label/main.py"
    config_file = model_folder + 'config.txt'

    
    with open(config_file, 'w') as f:
        config = '''
        train_dir={}
        dev_dir={}
        test_dir={}

        model_dir={}mod
        word_emb_dir={}

        norm_word_emb=False
        norm_char_emb=False
        number_normalized=False
        seg=False
        word_emb_dim=100
        char_emb_dim=30

        ###NetworkConfiguration###
        use_crf=False
        use_char=True
        word_seq_feature=LSTM
        char_seq_feature=LSTM
        feature=[POS] emb_size=25
        use_features={}

        ###TrainingSetting###
        status=train
        optimizer=SGD
        iteration=20
        batch_size=8
        ave_batch_loss=True
        mode=tag

        ###Hyperparameters###
        cnn_layer=4
        char_hidden_dim=50
        hidden_dim=800
        dropout=0.5
        lstm_layer=2
        bilstm=True
        learning_rate={}
        lr_decay=0.05
        momentum=0.9
        l2=0
        gpu=True
        #clip=

        ###MTL setup###
        index_of_main_tasks=0
        tasks=1
        tasks_weights=1

        ###PathsToAdditionalScripts###
        gold_dev_dep={}
        '''
        f.write(config.format(
            train_seq, dev_seq, dev_seq, model_folder, embeddings, features, learning_rate, gold_file
            ) 
        )
    monitoring = model_folder + 'monitoring.txt'
    # Execute the main.py file and train the model
    execute_main = 'CUDA_VISIBLE_DEVICES="{}" python "{}" --config "{}" --monitoring "{}"'.format(
        device, main_py, config_file, monitoring
        )
    os.system(execute_main)
    
def conllu_to_seq(file, tags):
    """
    Transform a single conllu file into a named temporary file in seq format
    """    
    if tags == 'none':
        tags = 'upos'

    # Encode the files
    encoding_script = 'dep2label/encode_dep2labels.py'
    seq_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    encoding_command = 'python "{}" --input "{}" --output "{}"  --encoding "2-planar-brackets-greedy" --mtl "3-task" --tag "{}"'.format(
        encoding_script, file, seq_file.name, tags
    )
    os.system(encoding_command)
    return seq_file

def predict_tags(treebank, tags, gold_conllu_file, device=0, perturbation=0):
    """
    Predict the PoS tags of a test file using a trained tagger given the treebank and the perturbation percentage
    """
    model_folder = 'tagger_models/tags_models/'
    if tags == 'upos':
        model_folder += 'UPOS/' + treebank + '/'
    elif tags == 'xpos':
        model_folder += 'XPOS/' + treebank + '/'
    elif tags == 'feats':
        model_folder += 'FEATS/' + treebank + '/'
    
    model = model_folder + '/mod.model'
    model_dset = model_folder + '/mod.dset'

    ncrf_file = ('dep2label/main.py')
    
    test_seq_gold = conllu_to_seq(gold_conllu_file, tags)
    test_seq_pred = model_folder + '/seq_files/test_pred.seq'
    # Execute the main.py file and predict the tags
    config ='''### Decode ###
        status=decode
        raw_dir={}
        decode_dir={}
        dset_dir={}
        load_model_dir={}
    '''.format(test_seq_gold.name, test_seq_pred, model_dset, model)

    config_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(config_file.name, 'w') as f:
        f.write(config)
    os.system('python {} --config {}'.format(ncrf_file, config_file.name))
    
    # Add the parsing to the test.pred file
    # Read tags
    with open(test_seq_pred, 'r') as f:
        pred_tags = f.read().split('\n\n')
    with open(gold_conllu_file, 'r') as f:
        gold_conllu_text = f.read().split('\n\n')
    pred_conllu_text = []

    tag_dic = {'upos': 3, 'xpos': 4, 'feats': 5}
    if len(pred_tags) == len(gold_conllu_text):
        for i in range(len(pred_tags)):
            sentence_tags = pred_tags[i] # text of sentence i in seq format
            sentence_tags = sentence_tags.split('\n') # rows of sentence i in seq format
            sentence_tags = [item.split('\t') for item in sentence_tags]
            sentence_tags = [item[1] for item in sentence_tags if len(item)>1 and item[1] not in {'-BOS-', '-EOS-'}] 

            sentence_conllu = gold_conllu_text[i] # text of sentence i in conllu format
            sentence_conllu = sentence_conllu.split('\n') # rows of sentence i in conllu format
            # only consider lines that start with a number
            sentence_conllu = [line.split('\t') for line in sentence_conllu]
            sentence_conllu = [line for line in sentence_conllu if line[0].isdigit()]
            for j in range(len(sentence_tags)):
                line = sentence_conllu[j]
                line[tag_dic[tags]] = sentence_tags[j]
                sentence_conllu[j] = '\t'.join(line)

            # quick fix for weird problem I don't understand why's happening
            for j in range(len(sentence_conllu)):
                if type(sentence_conllu[j]) == list:
                    sentence_conllu[j] = '\t'.join(sentence_conllu[j])

            pred_conllu_text.append('\n'.join(sentence_conllu))
    
        pred_conllu_text = '\n\n'.join(pred_conllu_text)
        pred_conllu_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        with open(pred_conllu_file.name, 'w') as f:
            f.write(pred_conllu_text)
        
    else:
        print('Error: the number of sentences in the gold and predicted files is different')            

    pred_conllu_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(pred_conllu_file.name, 'w') as f:
        f.write(pred_conllu_text)

    # Note: we should maybe return the test_seq_pred as a temporary file

    return pred_conllu_file, test_seq_pred

def evaluate(treebank, tags, test_conllu, test_seq_pred, device=0):
    """
    Evaluate the accuracy of a PoS tagger using the predicted and original test files
    """
    model_folder = 'tagger_models/tags_models/'

    if tags == 'upos':
        model_folder += 'UPOS/' + treebank + '/'
    elif tags == 'xpos':
        model_folder += 'XPOS/' + treebank + '/'
    elif tags == 'feats':
        model_folder += 'FEATS/' + treebank + '/'
    
    model_folder = 'tagger_models/tags_models/'
    if tags == 'upos':
        model_folder += 'UPOS/' + treebank + '/'
    elif tags == 'xpos':
        model_folder += 'XPOS/' + treebank + '/'
    elif tags == 'feats':
        model_folder += 'FEATS/' + treebank + '/'
        
    # Load the gold tags from the test_conllu file
    
    try:
        with open(test_conllu.name, 'r') as f:
            list_gold = f.read().splitlines()
    except:
        with open(test_conllu, 'r') as f:
            list_gold = f.read().splitlines()
            
    tags_dic = {'upos': 3, 'xpos': 4, 'feats': 5}
    list_gold = [row.split('\t')[tags_dic[tags]] for row in list_gold if row.split('\t')[0].isdigit()]

    test_seq_pred = model_folder + '/seq_files/test_pred.seq'
    with open(test_seq_pred, 'r') as f:
        list_predicted = f.read().split('\n')
    list_predicted = [row.split('\t') for row in list_predicted if row.split('\t') != ['']]
    list_predicted = [row[1] for row in list_predicted if row[0] not in {'-EOS-', '-BOS-'}]

    try:
        accuracy = 100*accuracy_score(list_gold,list_predicted)
        print('Accuracy: {:.2f}%'.format(accuracy))

    except ValueError:
        print([item for item in list_gold if item not in list_predicted])
        print('Inconsistent number of samples between predicted and gold files')
        return None

    return accuracy

if __name__ == '__main__':
    print(evaluate('UD_English-EWT','upos'))