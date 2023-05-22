import sys
from gb import evaluate

device = sys.argv[1]
with open('errors.txt', 'w') as f:
    f.write('')

for tags in [
    'none',
    'upos', 
    'xpos', 
    'feats'
]:
    if tags == 'xpos':
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_English-EWT', 
            'UD_Finnish-TDT', 
            'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES'
        ]
    elif tags == 'feats':
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 
            'UD_Finnish-TDT', 
            'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Hungarian-Szeged',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 
            'UD_Turkish-Penn'
        ]
    else:
        treebanks = [
            'UD_Afrikaans-AfriBooms', 'UD_Basque-BDT', 'UD_English-EWT', 
            'UD_Finnish-TDT',
            'UD_German-GSD',
            'UD_Indonesian-GSD', 'UD_Irish-IDT', 'UD_Lithuanian-HSE', 'UD_Maltese-MUDT', 'UD_Hungarian-Szeged',
            'UD_Polish-LFG', 'UD_Spanish-AnCora', 'UD_Swedish-LinES', 'UD_Turkish-Penn'
        ]

    #for treebank in treebanks:
        #if tags != 'none':
        #    for perturbed_tags in [
        #        'unperturbed',
        #        'perturbed'
        #    ]:
        #        try:
        #            evaluate(treebank, tags, 0, perturbed_tags, device=device)
        #        except:
        #            with open('errors.txt', 'a') as f:
        #                f.write(f'{treebank} {tags} 0 {perturbed_tags} {device}\n')
        #else:
        #    try:
        #        evaluate(treebank, tags, 0, 'none', device=device)
        #    except:
        #        with open('errors.txt', 'a') as f:
        #            f.write(f'{treebank} {tags} 0 none {device}\n')
            

    for treebank in treebanks:
        for perturbation in [2.5, 7.5, 12.5, 17.5]:#[10,20,30,40,50,60,70,80,90,100]:
            for perturbed_tags in [
                'unperturbed',
                'perturbed'
            ]:
                if tags != 'none':
                    try:
                        evaluate(treebank, tags, perturbation, perturbed_tags, epochs=10, device=device)
                    except:
                        with open('errors.txt', 'a') as f:
                            f.write(f'{treebank} {tags} {perturbation} {perturbed_tags} {device}\n')
                
                else:
                    if perturbed_tags == 'unperturbed':
                        try:
                            evaluate(treebank, tags, perturbation, 'none', epochs=10, device=device)
                        except:
                            with open('errors.txt', 'a') as f:
                                f.write(f'{treebank} {tags} {perturbation} {perturbed_tags} {device}\n')