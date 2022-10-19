import os
from sl import prepare_data
from sklearn.metrics import accuracy_score
import tempfile


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
        if filename.endswith('train.seq'):
            train_seq = encoded_folder + filename

        elif filename.endswith('dev.seq'):
            dev_seq = encoded_folder + filename
        
        elif filename.endswith('test.seq'):
            test_seq = encoded_folder + filename
    
    #  Remove the last column of the seq files
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
    print(execute_main)
    os.system(execute_main)

def predict(treebank, tags, device=0, perturbation=0):
    """
    Predict the PoS tags of a test file using a trained tagger given the treebank and the perturbation percentage
    """
    if tags == 'none':
        model_folder = 'tagger_models/none_models/' + treebank + '/'
    else:
        model_folder = 'tagger_models/tags_models/'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'
    
    model = model_folder + '/mod.model'
    model_dset = model_folder + '/mod.dset'
    # Define the test.seq file 
    encoded_folder = model_folder + 'seq_files/'
    for filename in os.listdir(encoded_folder):
        if filename.endswith('test.seq'):
            test_seq = encoded_folder + filename

    ncrf_file = ('dep2label/main.py')
    test_pred = test_seq.replace('test.seq', 'test.pred')

    # Execute the main.py file and predict the tags
    config ='''### Decode ###
        status=decode
        raw_dir={}
        decode_dir={}
        dset_dir={}
        load_model_dir={}
    '''.format(test_seq, test_pred, model_dset, model)

    config_file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    with open(config_file.name, 'w') as f:
        f.write(config)
    
    os.system('python {} --config {}'.format(ncrf_file, config_file.name))
    
    #with open(test_pred, 'r') as f:
    #    pred = f.read().splitlines()
    #
    #for i in range(pred):
    #    if pred[i] == '-BOS-':
    #        pred[i] = '-BOS-\t-BOS-'
    #    elif pred[i] == '-EOS-':
    #        pred[i] = '-EOS-\t-EOS-'

    #with open(test_seq, 'r') as f:
    #    test = f.read().splitlines()
    
    #TODO: CONLLU PART

def evaluate(treebank, tags, device=0):
    """
    Evaluate the accuracy of a PoS tagger using the predicted and original test files
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
    
    model = model_folder + 'mod.model'
    model_dset = model_folder + 'mod.dset'
    encoded_folder = model_folder + 'seq_files/'
    for filename in os.listdir(encoded_folder):
        if filename.endswith('test.seq'):
            test_gold = encoded_folder + filename
    
    test_predicted = test_gold.replace('.seq', '.pred')

    predict(treebank, tags, device)

    # Separate the tags and create two lists
    with open(test_gold, 'r') as f:
        list_gold = f.read().split('\n')
    list_gold = [row.split('\t') for row in list_gold if row.split('\t') != ['']]
    list_gold = [row[1] for row in list_gold if row[1] not in {'-EOS-', '-BOS-'}]

    with open(test_predicted, 'r') as f:
        list_predicted = f.read().split('\n')
    list_predicted = [row.split('\t') for row in list_predicted if row.split('\t') != ['']]
    list_predicted = [row[1] for row in list_predicted if row[1] not in {'-EOS-', '-BOS-'}]

    try:
        accuracy = 100*accuracy_score(list_gold,list_predicted)
    except ValueError:
        print([item for item in list_gold if item not in list_predicted])
        print('Inconsistent number of samples between predicted and gold files in {}'.format(treebank))
        return None
    
    with open('test_accuracy.txt', 'a') as f:
        f.write('{},{}: {}'.format(treebank, tags, accuracy, 2))
        f.write('\n')

    return accuracy

def avg_accuracy(treebanks, tags, perturbation=0, device=0):
    """
    Calculate the average accuracy of a PoS tagger over all treebanks
    """
    accuracy = []
    for treebank in treebanks:
        accuracy.append(evaluate(treebank, tags, device))

    accuracy = [float(item) for item in accuracy if item is not None]
    avg = round(sum(accuracy)/len(accuracy), 2)

    with open('test_accuracy.txt', 'a') as f:
        f.write('Average {}: {}'.format(tags, avg, 2))
        f.write('\n')
        f.write('\n')

    return avg

if __name__ == '__main__':
    treebanks = [
        'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
        'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Hungarian-Szeged',
        'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
    ]
    print(avg_accuracy(treebanks, 'feats', 0, 0))

    treebanks = [
        'UD_Afrikaans-AfriBooms', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
        'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT',
        'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES'
    ]
    print(avg_accuracy(treebanks, 'xpos', 0, 0))

    treebanks = [
        'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 'UD_Finnish-TDT', 'UD_German-GSD',
        'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT', 'UD_Hungarian-Szeged',
        'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
    ]

    print(avg_accuracy(treebanks, 'upos', 0, 0))