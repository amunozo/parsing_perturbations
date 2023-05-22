import numpy as np
import os
import tempfile
from perturbate import perturbate_file
import tagger


ud_folder = '/home/alberto/Universal Dependencies 2.9/ud-treebanks-v2.9/'

def train(treebank, tags, device):
    """
    Train a Pointer network parser using a UD treebank
    """
    # ¿Hace falta utilizar CoNLLx o CoNLLu?
    if tags == 'none':
        model_folder = 'pointer_models/none_models/' + treebank + '/'
    
    else:
        model_folder = 'pointer_models/tags_models/'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
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
    if tags == 'none':
        config = 'SyntacticPointer/experiments/configs/parsing/l2r_none.json'
    else:
        config = 'SyntacticPointer/experiments/configs/parsing/l2r.json'
    # Train the model; it should let you differentiate between using the different tags and no tags
    train_command = 'CUDA_VISIBLE_DEVICES={} OMP_NUM_THREADS=4 python -u "SyntacticPointer/experiments/parsing.py" \
                    --mode train --config "{}" --num_epochs 100 --batch_size 32 \
                    --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
                    --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
                    --word_embedding fasttext --word_path "{}" --char_embedding random --tags "{}"\
                    --punctuation "." "``" "''" ":" "," \
                    --train "{}" \
                    --dev "{}" \
                    --test "{}" \
                    --model_path "{}"'.format(device, config, embeddings, tags, train_conllx, dev_conllx, dev_conllx, model_folder + 'model')
    
    os.system(train_command)

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

def evaluate(treebank, tags, perturbation, perturbed_tagging, epochs=10, device=0):
    if tags == 'none':
        model_folder = 'pointer_models/none_models/' + treebank + '/'
    
    else:
        model_folder = 'pointer_models/tags_models/'
        if tags == 'upos':
            model_folder += 'UPOS/' + treebank + '/'
        elif tags == 'xpos':
            model_folder += 'XPOS/' + treebank + '/'
        elif tags == 'feats':
            model_folder += 'FEATS/' + treebank + '/'

    # Select train and dev file
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

    if tags == 'none':
        config = 'SyntacticPointer/experiments/configs/parsing/l2r_none.json'
    else:
        config = 'SyntacticPointer/experiments/configs/parsing/l2r.json'

    if perturbation != 0:
        for i in range(epochs):
            if tags != 'none':
                # Predict PoS tags; it should give a list with the list of predicted tags to insert in the conllx file
                if perturbed_tagging == 'perturbed':
                    perturbed_test_conllu = perturbate_file(treebank, perturbation)
                    predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, perturbed_test_conllu.name)
                else:
                    predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, gold_conllu)
                    predicted_test_conllu = perturbate_file(treebank, perturbation, predicted_test_conllu.name)

                acc = tagger.evaluate(treebank, tags, gold_conllu, test_seq_pred)

                # Measure tagger accuracy
                acc_list.append(acc)

                perturbed_text_conllx = conllu_to_conll(predicted_test_conllu.name, None, test=True)
            
            else:
                perturbed_test_conllu = perturbate_file(treebank, perturbation)
                perturbed_text_conllx = conllu_to_conll(perturbed_test_conllu.name, None, test=True)
        

            # Get UAS and LAS
            evaluate_command = 'CUDA_VISIBLE_DEVICES={} OMP_NUM_THREADS=4 python -u "SyntacticPointer/experiments/parsing.py" \
                            --mode parse --config "{}" --num_epochs 100 --batch_size 256 \
                            --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
                            --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
                            --word_embedding fasttext --char_embedding random --tags "{}"\
                            --punctuation "." "``" "''" ":" "," \
                            --test "{}" \
                            --model_path "{}"'.format(device, config, tags, perturbed_text_conllx, model_folder + 'model')
            print(evaluate_command)
            
            # TODO: save UAS and LAS
            output = os.popen(evaluate_command).read()
            output = output.replace('(', '').replace(')', '').replace('\n', '').split(', ')
            UAS, LAS = float(output[0]), float(output[1])

            uas_list.append(UAS)
            las_list.append(LAS)
            print('UAS: ', UAS, 'LAS: ', LAS)

    else:
        epochs = 1
        if tags != 'none':
            predicted_test_conllu, test_seq_pred = tagger.predict_tags(treebank, tags, gold_conllu)
            acc = tagger.evaluate(treebank, tags, gold_conllu, test_seq_pred)
            acc_list.append(acc)
        
        else:
            acc = 100
            acc_list.append(acc)
            predicted_test_conllu = gold_conllu
        
        
        
        # how's the output here????????

        try:
            perturbed_test_conllx = conllu_to_conll(predicted_test_conllu, None, test=True)
            
        except:
            perturbed_test_conllx = conllu_to_conll(predicted_test_conllu.name, None, test=True)
        
        # Get UAS and LAS
        evaluate_command = 'CUDA_VISIBLE_DEVICES={} OMP_NUM_THREADS=4 python -u "SyntacticPointer/experiments/parsing.py" \
                        --mode parse --config "{}" --num_epochs 100 --batch_size 256 \
                        --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
                        --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
                        --word_embedding fasttext --char_embedding random --tags "{}"\
                        --punctuation "." "``" "''" ":" "," \
                        --test "{}" \
                        --model_path "{}"'.format(device, config, tags, perturbed_test_conllx, model_folder + 'model')
        
        output = os.popen(evaluate_command).read()
        output = output.replace('(', '').replace(')', '').replace('\n', '').split(', ')
        UAS, LAS = float(output[0]), float(output[1])

        uas_list.append(UAS)
        las_list.append(LAS)
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
            treebank, 'TB', tags, perturbed_tagging, perturbation, average_uas, std_uas, average_las, std_las, average_acc, std_acc, epochs
        )
    )
            
# OLD TORCH VERSION WAS 1.7.1 -> TRYING TO UPDATE IT TO 2.0.1


if __name__ == '__main__':
    evaluate(
        treebank='UD_Irish-IDT',
        tags='upos',
        perturbed_tagging='unperturbed',
        perturbation=11,
        epochs=1,
        device=0
    )