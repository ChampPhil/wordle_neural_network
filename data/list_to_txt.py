from wordle_secret_words import get_secret_words
from valid_wordle_guesses import get_valid_wordle_guesses

with open('wordle_valid_guesses.txt', 'w') as f:
    word_str = ""
    word_set = get_valid_wordle_guesses()
    for i, word in enumerate(list(word_set)):
        word_str += f'{word}'
        if i+1 < len(word_set):
            word_str += '\n'
    f.write(word_str)