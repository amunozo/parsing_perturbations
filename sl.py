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
        if filename.endswith('train.conllu'):
            train_conllu = ud_folder + treebank + '/' + filename
            train = filename
            print(filename)
            files.append((train_conllu, train))
        elif filename.endswith('dev.conllu'):
            dev_conllu = ud_folder + treebank + '/' + filename
            dev = filename
            files.append((dev_conllu, dev))
        elif filename.endswith('test.conllu'):
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
    

def train(treebank, tags, device, encoding='2-planar-brackets-greedy', learning_rate = 0.02):
    """
    Train a SL dependency parser using a UD treebank
    """
    if tags == 'none':
        model_folder = 'sl_models/none_models/' + treebank + '/'
        features = 'False'
    
    else:
        model_folder = 'sl_models/tags_models/'
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
        iteration=50
        batch_size=8
        ave_batch_loss=True
        mode=parse

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
        index_of_main_tasks=0,1,2,3
        tasks=3
        tasks_weights=1|1|1
        encoding=2-planar-brackets-greedy

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


if __name__ == '__main__':
    train('UD_Lithuanian-HSE', 'feats', 0)