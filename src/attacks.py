import string
from nltk.corpus import stopwords as SW
from collections import defaultdict
import random
from random import shuffle
import numpy as np

keyboard_mappings = None


def get_keyboard_neighbors(ch, treebank):
    global keyboard_mappings
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = get_keyboard(treebank)
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return [ch]
    return keyboard_mappings[ch]

def get_keyboard(treebank):
    """
    Returns a list with the keyboard rows for the language of each treebank
    """
    # Each language has a specific keyboard including the diacritics of the language
    keyboard = {
        'UD_Afrikaans-AfriBooms': ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***'],
        'UD_Spanish-AnCora': ['qwertyuiop', 'asdfghjklñ', 'zxcvbnm***'],
        'UD_Basque-BDT': ['qwertyuiop', 'asdfghjklñ', 'zxcvbnm***'],
        'UD_English-EWT': ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***'],
        'UD_Finnish-TDT': ['qwertyuiopå', 'asdfghjklöä', 'zxcvbnm****'],
        'UD_German-GSD': ['qwertzuiopü', 'asdfghjklöä', 'yxcvbnm****'],
        'UD_Indonesian-GSD': ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***'],
        'UD_Irish-IDT': ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***'],
        'UD_Lithuanian-HSE': ['qwertyuiop', 'asdfghjkl*', 'zxcvbnm***'],
        'UD_Maltese-MUDT': ['qwertyuiopġħ', 'asdfghjkl**ż', 'zxcvbnm****'],
        'UD_Hungarian-Szeged': ['qwertzuiopőú', 'asdfghjkléáű', 'yxcvbnm****'],
        'UD_Polish-LFG': ['qwertyuiopżś', 'asdfghjklłóą', 'zxcvbnm*****'],
        'UD_Swedish-LinES': ['qwertyuiopå', 'asdfghjklöä', 'zxcvbnm****'],
        'UD_Turkish-Penn': ['qwertyuiopğü', 'asdfghjklşöç', 'zxcvbnm*****']
    }
    return keyboard[treebank]

def get_alphabet(treebank):
    """
    Returns a list with the alphabet for the language of each treebank
    """
    alphabet = {
        'UD_Afrikaans-AfriBooms': 'abcdefghijklmnopqrstuvwxyz',
        'UD_Spanish-AnCora': 'abcdefghijklmnñopqrstuvwxyz',
        'UD_Basque-BDT': 'abcdefghijklmnñopqrstuvwxyz',
        'UD_English-EWT': 'abcdefghijklmnopqrstuvwxyz',
        'UD_Finnish-TDT': 'abcdefghijklmnopqrstuvwxyzåäö',
        'UD_German-GSD': 'abcdefghijklmnopqrstuvwxyzüöä',
        'UD_Indonesian-GSD': 'abcdefghijklmnopqrstuvwxyz',
        'UD_Irish-IDT': 'abcdefghijklmnopqrstuvwxyz',
        'UD_Lithuanian-HSE': 'abcdefghijklmnopqrstuvwxyz',
        'UD_Maltese-MUDT': 'abcdefghijklmnopqrstuvwxyzġħż',
        'UD_Hungarian-Szeged': 'abcdefghijklmnopqrstuvwxyzáéíóöőúüű',
        'UD_Polish-LFG': 'abcdefghijklmnopqrstuvwxyząćęłńóśźż',
        'UD_Swedish-LinES': 'abcdefghijklmnopqrstuvwxyzåäö',
        'UD_Turkish-Penn': 'abcdefghijklmnopqrstuvwxyzğüşöç'
    }

    return alphabet[treebank]    
    
def is_valid_attack(line, char_idx):
    line = line.lower()
    if char_idx == 0 or char_idx == len(line) - 1:
        # first and last chars of the sentence
        return False
    if line[char_idx-1] == ' ' or line[char_idx+1] == ' ':
        # first and last chars of the word
        return False

    return True


def get_random_attack(line, treebank):
    num_chars = len(line)
    NUM_TRIES = 10

    for _ in range(NUM_TRIES):
        char_idx = np.random.choice(range(num_chars), 1)[0]
        if is_valid_attack(line, char_idx):
            attack_type = ['swap', 'drop', 'add', 'key']
            #attack_probs = np.array([1.0,1.0,1.0,1.0]) # equal prob
            attack_probs = np.array([1.0, 1.0, 10.0, 2.0]) # original prob (more similar results to the first paper xd)
            attack_probs = attack_probs/sum(attack_probs)
            attack = np.random.choice(attack_type, 1, p=attack_probs)[0]
            if attack == 'swap':
                return line[:char_idx] + line[char_idx:char_idx+2][::-1] + line[char_idx+2:]
            elif attack == 'drop':
                return line[:char_idx] + line[char_idx+1:]
            elif attack == 'key':
                sideys = get_keyboard_neighbors(line[char_idx], treebank)
                new_ch = np.random.choice(sideys, 1)[0]
                return line[:char_idx] + new_ch + line[char_idx+1:]
            else: # attack type is add
                alphabets = get_alphabet(treebank)
                alphabets = [ch for ch in alphabets]
                new_ch = np.random.choice(alphabets, 1)[0]
                return line[:char_idx] + new_ch + line[char_idx:]
        
    return line