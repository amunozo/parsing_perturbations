import os


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
    # Train the model; it should let you differentiate between using the different tags and no tags
    train_command = 'CUDA_VISIBLE_DEVICES={} OMP_NUM_THREADS=4 python -u "SyntacticPointer/experiments/parsing.py" \
                    --mode train --config "SyntacticPointer/experiments/configs/parsing/l2r.json" --num_epochs 100 --batch_size 32 \
                    --opt adam --learning_rate 0.001 --lr_decay 0.999997 --beta1 0.9 --beta2 0.9 --eps 1e-4 --grad_clip 5.0 \
                    --loss_type token --warmup_steps 40 --reset 20 --weight_decay 0.0 --unk_replace 0.5 --beam 10 \
                    --word_embedding fasttext --word_path "{}" --char_embedding random --tags "{}"\
                    --punctuation "." "``" "''" ":" "," \
                    --train "{}" \
                    --dev "{}" \
                    --test "{}" \
                    --model_path "{}"'.format(device, embeddings, tags, train_conllx, dev_conllx, dev_conllx, model_folder + 'model')

    os.system(train_command)

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

if __name__ == '__main__':
    train('UD_English-EWT', 'none', 0)