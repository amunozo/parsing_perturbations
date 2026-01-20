import os
import numpy as np
import tempfile
import src.tagger as tagger
import src.const as const
from src.perturbate import perturbate_file
from src.utils import prepare_data

ud_folder = const.UD_FOLDER
    
def train(treebank, tags, device, encoding='2-planar-brackets-greedy', epochs=10, learning_rate = 0.02):
    """
    Train a SL dependency parser using a UD treebank
    """
    if tags == 'none':
        model_folder = os.path.join(const.SL_MODELS, 'none_models', treebank) + '/'
        features = 'False'
    
    else:
        model_folder = os.path.join(const.SL_MODELS, 'tags_models')
        features = 'True'
        if tags == 'upos':
            model_folder = os.path.join(model_folder, 'UPOS', treebank) + '/'
        elif tags == 'xpos':
            model_folder = os.path.join(model_folder, 'XPOS', treebank) + '/'
        elif tags == 'feats':
            model_folder = os.path.join(model_folder, 'FEATS', treebank) + '/'
        
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
    embeddings = os.path.join(const.EMBEDDINGS_FOLDER, language)

    # Create the config file
    # Now we localize where the main.py file is to train the model
    main_py = os.path.join(const.DEP2LABEL_PATH, "main.py")
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
        index_of_main_tasks=0,1,2
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

def evaluate(treebank, tags, perturbation, perturbed_tagging, epochs=10, encoding='2-planar-brackets-greedy', device=0):

    """
    Evaluate a SL parser on the perturbed test set of a treebank using a certain percentage of
    perturbed words and averaging the result over a certain number of epochs
    """
    # Locate test file
    if tags == 'none':
        model_folder = 'sl_models/none_models/' + treebank + '/'
    
    else:
        model_folder = 'sl_models/tags_models/'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'
    
    model = model_folder + 'mod'    
    encoded_folder = model_folder + 'seq_files/'

    for filename in os.listdir(ud_folder + treebank + '/'):
        if filename.endswith('ud-test.conllu'):
            gold_conllu = ud_folder + treebank + '/' + filename

    #for filename in os.listdir(encoded_folder):
    #    if filename.endswith('ud-test.seq'):
    #        test_seq_gold = encoded_folder + filename
    

    ncrf_dir = const.DEP2LABEL_PATH
    test_seq_pred = model_folder + '/seq_files/test_pred.seq'

    # Perturbate the test set
    uas_list = []
    las_list = []
    acc_list = []

    if perturbation != 0:
        for i in range(epochs):
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
            
            else:
                predicted_test_conllu = perturbate_file(treebank, perturbation)
                acc = 100
            
            test_seq_perturbed = tagger.conllu_to_seq(predicted_test_conllu.name, tags)

            # Get UAS and LAS
            output_conllu = tempfile.NamedTemporaryFile(mode='w', delete=False)
            output_seq = tempfile.NamedTemporaryFile(mode='w', delete=False)

            decode_py = os.path.join(const.DEP2LABEL_PATH, 'decode.py')
            execute_decode = 'python  "{}" --test "{}" --gold "{}" --model "{}" --gpu True \
                --output "{}" --encoding "{}" --ncrf "{}" --decode "{}"'.format(
                    decode_py,
                    test_seq_perturbed.name,
                    predicted_test_conllu.name,
                    model,
                    output_seq.name,
                    encoding,
                    ncrf_dir,
                    output_conllu.name
            )

            # Get the score output
            score = os.popen(execute_decode).read()

            UAS = score.split('\t')[0]
            LAS = score.split('\t')[1].replace('\n', '')
            uas_list.append(float(UAS))
            las_list.append(float(LAS))
            acc_list.append(acc)
            print('UAS: ', UAS, 'LAS: ', LAS, 'ACC: ', acc)

    else:
        epochs = 1
        if tags != 'none':
            predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, gold_conllu)
            acc = tagger.evaluate(treebank, tags, gold_conllu, test_seq_pred)
            acc_list.append(acc)
        else:
            acc = 100
            acc_list.append(acc)
            # predicted_test_conllu = gold_conllu
            # pass into a temp file to get homogeinity
            predicted_test_conllu = tempfile.NamedTemporaryFile(mode='w', delete=False)
            with open(gold_conllu, 'r') as f:
                predicted_test_conllu.write(f.read())


        test_seq_perturbed = tagger.conllu_to_seq(predicted_test_conllu.name, tags)
        
        # Get UAS and LAS
        output_conllu = tempfile.NamedTemporaryFile(mode='w', delete=False)
        output_seq = tempfile.NamedTemporaryFile(mode='w', delete=False)
        
        decode_py = os.path.join(const.DEP2LABEL_PATH, 'decode.py')
        execute_decode = 'python  "{}" --test "{}" --gold "{}" --model "{}" --gpu True \
            --output "{}" --encoding "{}" --ncrf "{}" --decode "{}"'.format(
                decode_py,
                test_seq_perturbed.name,
                predicted_test_conllu.name,
                model,
                output_seq.name,
                encoding,
                ncrf_dir,
                output_conllu.name
        )

        # Get the score output
        score = os.popen(execute_decode).read()
        with open(output_conllu.name, 'r') as f:
            print(f.read()[:100])

        UAS = score.split('\t')[0]
        LAS = score.split('\t')[1].replace('\n', '')

        uas_list.append(float(UAS))
        las_list.append(float(LAS))
        acc_list.append(acc)
        print('UAS: ', UAS, 'LAS: ', LAS, 'ACC: ', acc)      

    average_uas = sum(uas_list) / len(uas_list)
    average_las = sum(las_list) / len(las_list)
    average_acc = sum(acc_list) / len(acc_list)
    std_uas = np.std(uas_list)
    std_las = np.std(las_list)
    std_acc = np.std(acc_list)

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
    csv_file = const.SCORES_FILE
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('Treebank,Parser,Tags,Perturbed tagging,Perturbation,UAS,UAS std,LAS,LAS std,Tagger acc,Tagger acc std,Epochs\n')

    with open(csv_file, 'a') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            treebank, 'SL', tags, perturbed_tagging, perturbation, average_uas, std_uas, average_las, std_las, average_acc, std_acc, epochs
        )
    )
if __name__ == '__main__':
    for tags in ['none', 'upos', 'xpos', 'feats']:
        if tags == 'xpos':
            treebanks = [
                'UD_Afrikaans-AfriBooms',
                'UD_English-EWT', 
                'UD_Finnish-TDT', 
                'UD_German-GSD',
                'UD_Indonesian-GSD', 
                'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT',
                'UD_Polish-LFG', 
                'UD_Spanish-AnCora', 'UD_Swedish-LinES'
    ]
        elif tags == 'feats':
            treebanks = [
                'UD_Afrikaans-AfriBooms',
                'UD_Basque-BDT', 'UD_English-EWT', 
                'UD_Finnish-TDT', 'UD_German-GSD',
                'UD_Indonesian-GSD',
                'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Hungarian-Szeged',
                'UD_Polish-LFG',
                'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
            ]
        else:
            treebanks = [
                'UD_Afrikaans-AfriBooms', 
                'UD_Basque-BDT', 
                'UD_English-EWT', 
                'UD_Finnish-TDT', 'UD_German-GSD',
                'UD_Indonesian-GSD', 
                'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT', 'UD_Hungarian-Szeged',
                'UD_Polish-LFG',
                'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
    ]
        for treebank in treebanks:        
            for perturbation in range(0, 101, 10):
                evaluate(treebank, tags, perturbation, 'perturbed', epochs=1)
