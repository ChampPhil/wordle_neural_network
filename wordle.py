#!pip3 install stable-baselines3 shimmy>=2.0 torch sb3-contrib

import random
from collections import Counter
from colorama import Fore, Back, Style, init
from enum import Enum
import time
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import os
import string
from collections import Counter
from typing import Optional, Callable
from sb3_contrib import MaskablePPO

# init(autoreset=True) #Ends color formatting after each print statement
from wordle_secret_words import get_secret_words
from valid_wordle_guesses import get_valid_wordle_guesses
from wordle_utils import _get_feedback_str as get_feedback
from wordle_utils import _convert_word_to_array, _convert_array_to_word, _convert_feedback_str_to_feedback_array
from wordle_env import WordleEnv

MAX_GUESSES = 6

# Convert Data to NP Arrays (for neural network)
print('Converting data to np.ndarrays for the nerual network...')
secret_word_arrays = [_convert_word_to_array(word) for word in get_secret_words()]
valid_guess_arrays = [_convert_word_to_array(word) for word in get_valid_wordle_guesses()]

# Get Neural Network
print("Loading AI model...")
model = MaskablePPO.load("model")
env = WordleEnv(secret_word_arrays=secret_word_arrays, max_guesses=MAX_GUESSES)
print("AI Model loaded!")

def display_board(guesses: list[str], feedbacks: list[str]):
    """
    Args:
        guesses (List[str]): A list of string guesses, which could be empty
        feedbacks: (List[str]): A list of feedback strings, which could be empty
           - Each feedback is in this format: '-e-E-'. A dash means that the letter is not in the word, a lowercase means the letter
           is in the word but in the wrong position, and a capital letter means the word is in the right position / and is in the word
    """
    for guess, feedback in list(zip(guesses, feedbacks)): # Iterate over each (guess, feedback pair)
        for feedback_index, feedback_letter in enumerate(feedback):
            if feedback_letter.lower() == '-':
                print(Back.LIGHTBLACK_EX + guess[feedback_index], end='')
            elif feedback_letter.islower():
                print(Back.YELLOW + guess[feedback_index], end='')
            elif feedback_letter.isupper():
                print(Back.GREEN + guess[feedback_index], end='')
            
        print(Style.RESET_ALL, end='')
        print()


"""
def get_AI_guess(guesses: list[str], feedbacks: list[str], secret_words: set[str], valid_guesses: set[str]) -> str:
    'Analyzes feedback from previous guesses/feedback (if any) to make a new guess
        
        Args:
         guesses (list): A list of string guesses, which could be empty
         feedbacks (list): A list of feedback strings, which could be empty
         secret_words (set): A set of potential secret words
         valid_guesses (set): A set of valid AI guesses
        
        Returns:
         str: a valid guess that is exactly 5 uppercase letters
    '
    ### END SOLUTION 
    if not feedbacks or not guesses:
        return 'SALET'
    
    if len(guesses) == 1 and len(feedbacks) == 1:
        return 'FERMS'

    #print(f"Initial secret-words (length of {len(secret_words)}) {list(secret_words)[:20]}")
    secret_words_copy = secret_words.copy()
    secret_words_to_remove = set()

    incorrect_letters = set() # Letters that the final word CANNOT have ANYWHERE
    incorrect_letters_and_positions = [] # Letters that the final word CANNOT have in certain positions
    must_have_letters = set() # Letters that the final word MUST have ANYWHERE
    must_have_letters_and_positions = [] # Letters that the final word MUST have in certain positions

    # Find letters that must be in final word
    for guess, feedback in list(zip(guesses, feedbacks)):
        for i, (guess_char, feedback_char) in enumerate(list(zip(guess, feedback))):
            if feedback_char.islower():
                must_have_letters.add(guess_char)

            if feedback_char.isupper():
                must_have_letters_and_positions.append((guess_char, i))

    # Find letters that should not be used ANYWHERE, and letters that should not be used in SPECIFIC POSITIONS
    for guess, feedback in list(zip(guesses, feedbacks)):
        for i, (guess_char, feedback_char) in enumerate(list(zip(guess, feedback))):
            if feedback_char.lower() == '-' and guess_char not in must_have_letters: 
                incorrect_letters.add(guess_char)

            if feedback_char.islower() and (guess_char.lower() == feedback_char.lower()):
                incorrect_letters_and_positions.append((guess_char, i))

    # Remove previous guesses
    for secret_word in secret_words_copy:
        if secret_word.upper() in guesses:
            secret_words_to_remove.add(secret_word)
    
    
    # Remove words with letters that should not be used at all
    for secret_word in secret_words_copy:
        for incorrect_letter in incorrect_letters:
            if incorrect_letter in secret_word:
                secret_words_to_remove.add(secret_word)
    
    # Remove words with letters used in the wrong spot
    for secret_word in secret_words_copy:
        for incorrect_letter, incorrect_letter_position in incorrect_letters_and_positions:
            if incorrect_letter in secret_word and (secret_word[incorrect_letter_position] == incorrect_letter):
               secret_words_to_remove.add(secret_word)

    # Remove words that dont have letters that must be used
    for secret_word in secret_words_copy:
        for must_have_letter in must_have_letters:
            if must_have_letter not in secret_word:
                secret_words_to_remove.add(secret_word)

        for must_have_letter, must_have_letter_position in must_have_letters_and_positions:
            if must_have_letter not in secret_word or secret_word[must_have_letter_position] != must_have_letter:
                secret_words_to_remove.add(secret_word)
        

    for word_to_remove in secret_words_to_remove:
        try:
            secret_words_copy.remove(word_to_remove)
        except:
            pass 

    #print(f"Final secret-words (length of {len(secret_words_copy)}): {list(secret_words_copy)[:20]}")
    return random.choice(list(secret_words_copy or secret_words))
"""



def get_AI_guess(guesses: list[str], feedbacks: list[str], secret_words: set[str], valid_guesses: set[str]) -> str:
    """
    Analyzes feedback from previous guesses/feedback (if any) to make a new guess using a neural network.
        
        Args:
         guesses (list): A list of string guesses, which could be empty
         feedbacks (list): A list of feedback strings, which could be empty
         secret_words (set): A set of potential secret words (unusued, meant to maintain function signature consistency)
         valid_guesses (set): A set of valid AI guesses (unusued, meant to maintain function signature consistency)
        
        Returns:
         str: a valid guess that is exactly 5 uppercase letters
    """
    # Reset environment, and give it the current guesses / feedbacks history
    obs, _ = env.reset(options={
            'guesses': [_convert_word_to_array(guess) for guess in guesses],
            'feedbacks': [_convert_feedback_str_to_feedback_array(guess, feedback) for guess, feedback in list(zip(guesses, feedbacks))]
        }
    )
    
    # Get AI guess (with output masking)
    mask = env.valid_action_mask()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    action_int = int(action)

    # Get AI guess as an np.ndarray and convert to a word
    guess_array = env._action_to_word_array[action_int]
    return _convert_array_to_word(guess_array).upper()

class PotentialGameModes(Enum):
    AI_MODE='ai_mode',
    HUMAN_MODE='human_mode'

def get_mode() -> PotentialGameModes:
    """
    Uses input() to get the mode
    """
    game_mode = input("What is the type of game that you want to play (enter 1 for manual mode, and 2 for AI mode): ").strip()
    try:
        game_mode_int = int(game_mode)
        if game_mode_int not in [1, 2]:
            print(Fore.RED + "Please input either 1 and 2")
            print(Style.RESET_ALL, end='')
            get_mode()
        
        return PotentialGameModes.HUMAN_MODE if game_mode_int == 1 else PotentialGameModes.AI_MODE
    except:
        print(Fore.RED + "Please input only the values 1 or 2 (no other characters)")
        print(Style.RESET_ALL, end='')
        get_mode()
        
            

def get_guess():
    """
    Use input() to get a user guess
    """

    user_guess = input("What is your current guess?: ")
    if type(user_guess) is not str:
       print(Fore.RED + "Please input a valid value")
       print(Style.RESET_ALL, end='')
       return get_guess()
   
    if len(user_guess.strip()) != 5:
       print(Fore.RED + "Your guess must be five letters")
       print(Style.RESET_ALL, end='')
       return get_guess()

    if user_guess.strip().upper() not in get_valid_wordle_guesses():
        print(Fore.RED + "Your guess must be a valid word")
        print(Style.RESET_ALL, end='')
        return get_guess()
    
    return user_guess.strip().upper()

def is_guess_correct(guess: str, secret: str):
    guess, secret = guess.strip(), secret.strip()
    return guess.lower() == secret.lower()
   
   
def main():
    print("Welcome to Wordle!")
    game_mode = get_mode()
    print(f"You have {MAX_GUESSES} gueeses to guess a given word! GO!\n")
   
    # Track game state
    guesses = []
    feedbacks = []
    num_guesses_left = MAX_GUESSES
    did_win = False
    secret_word = random.choice(list(get_secret_words()))

    print(f"Secret word: {secret_word}")
   
    while len(guesses) < MAX_GUESSES:
        if game_mode == PotentialGameModes.HUMAN_MODE:
            user_guess = get_guess()
        else:
            user_guess = get_AI_guess(
                guesses=list(guesses),
                feedbacks=list(feedbacks),
                secret_words=get_secret_words(),
                valid_guesses=get_valid_wordle_guesses()
            )
            time.sleep(0.5)
            print(f"The AI guessed the following word: {user_guess}")

        user_guess_feedback = get_feedback(user_guess, secret_word)

        if is_guess_correct(user_guess, secret_word):
            print("You correctly guessed the word!! Congrats.")
            did_win = True
            break

        guesses.append(user_guess)
        feedbacks.append(user_guess_feedback)
        print()
        display_board(guesses, feedbacks)
        #print(f"Feedback for your guess: {user_guess_feedback}")

        num_guesses_left -= 1
        print(f"You have {num_guesses_left} guesses left.")

    if did_win:
        print("YOU WON!!")
    else:
        print(f"YOU LOST!! The word was {secret_word}")
   




if __name__ == "__main__":
    main()
