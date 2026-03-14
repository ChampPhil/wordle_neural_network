import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import os
import string
from collections import Counter
from typing import Optional, Callable

CHARACTERS_TO_NUMS_DICT: dict[str, int] = dict(zip(string.ascii_lowercase, range(1, 27)))
NUMS_TO_CHARACTERS_DICT: dict[int, str] = {value: key for key, value in CHARACTERS_TO_NUMS_DICT.items()}

def _convert_word_to_array(word: str) -> np.array:
  """
  Converts (case-insensitive) a N-character word to a numeric np.array of shape (N, )
  Args:
    word (str): The word to convert to a numeric array
  Returns:
    np.array: A numerical representation of the word

  Examples
    >>> _convert_word_to_array("lever")
    array([12,  5, 22,  5, 18])

    >>> _convert_word_to_array("EATEN")
    array([ 5,  1, 20,  5, 14])
  """
  return np.array([CHARACTERS_TO_NUMS_DICT[char] for char in word.lower()])

def _convert_array_to_word(array: np.array) -> str:
  """
  Converts a numeric np.array of shape (N, ) to a string with N characters
  Args:
    array (np.array): A numerical representation of the word
  Returns:
    str: A string representation of the word

  Examples
    >>> _convert_array_to_word(np.array([12, 5, 22, 5, 18]))
    'lever'

    >>> _convert_array_to_word(np.array([5, 1, 20, 5, 14]))
    'eaten'
  """
  return "".join([NUMS_TO_CHARACTERS_DICT[num] for num in array])

def _get_feedback_str(guess: str, secret_word: str) -> str:
    '''Generates a feedback string based on comparing a 5-letter guess with the secret word.
       The feedback string uses the following schema:
        - Correct letter, correct spot: uppercase letter ('A'-'Z')
        - Correct letter, wrong spot: lowercase letter ('a'-'z')
        - Letter not in the word: '-'

        Args:
            guess (str): The guessed word
            secret_word (str): The secret word

        Returns:
            str: Feedback string, based on comparing guess with the secret word

        Examples
        >>> get_feedback("lever", "EATEN")
        '-e-E-'

        >>> get_feedback("LEVER", "LOWER")
        'L--ER'

        >>> get_feedback("MOMMY", "MADAM")
        'M-m--'

        >>> get_feedback("ARGUE", "MOTTO")
        '-----'
    '''
    guess, secret_word = guess.lower(), secret_word.lower()
    result = ['', '', '', '', '']

    secret_word_counter = dict(Counter(secret_word))
    for i in range(0, 5): # Loop over each letter
        if guess[i] == secret_word[i]:
            letter = guess[i] # Get current letter
            result[i] = letter.upper()

            if letter in secret_word_counter and secret_word_counter[letter] > 0:
                secret_word_counter[letter] = secret_word_counter[letter] - 1

    for i in range(0, 5): # Loop over each letter
        if not result[i]:
            letter = guess[i]  # Get current letter
            if letter in secret_word_counter and secret_word_counter[letter] > 0:
                result[i] = letter.lower()
                secret_word_counter[letter] = secret_word_counter[letter] - 1

    # Make everything else '-'
    result = [char if char else '-' for char in result]
    return ''.join(result)

def _convert_feedback_str_to_feedback_array(guess_str: str, feedback_str: str) -> np.array:
    """
    Convert a feedback string (like '-e-E-' to an np.ndarray)
    Args:
        guess_str (str): The guess (as a string)
        feedback_str (str): The feedback (as a string)
    Returns:
        np.ndarray: A numerical representation of the feedback in the shape of 5x2 
         - Each row is [numeric-encoding, 0-2], and there are 5 rows (one for each letter, in order)
         - The numeric encoding is 1-26, with 1 representing 'a', 2 representing 'b', etc.
         - The 0-2 represents the correctness of the letter. 0 means the letter is not in the actual word, 1 means
           that the letter is in the word but placed incorrectly, and 2 means the letter is in the word and in the
           correct position
    """
    res = []
    # Loop over feedback and create the array
    for i, feedback_char in enumerate(feedback_str):
        guess_char = guess_str[i].lower() # Get character from string representation for dict lookup
        if feedback_char == '-':
            res.append([CHARACTERS_TO_NUMS_DICT[guess_char], 0])
        if feedback_char.islower():
            res.append([CHARACTERS_TO_NUMS_DICT[guess_char], 1])
        if feedback_char.isupper():
            res.append([CHARACTERS_TO_NUMS_DICT[guess_char], 2])

    # Convert to an array
    return np.array(res)


def get_feedback_array(guess_array: np.array, secret_word_array: np.array) -> np.array:
  """
  Convert a guess to a numerical representation, as a np.array
  Args:
     guess_array (np.array): A numerical representation of the guessed word.
     secret_word_array (np.array): A numerical representation of the actual word.

  Returns:
      np.array: A 2D np.array representing that correctness of the guess. Each row is [numeric-encoding, 0-2], and there are 5 rows.
         - The numeric encoding is 1-26, with 1 representing 'a', 2 representing 'b', etc.
         - The 0-2 represents the correctness of the letter. 0 means the letter is not in the actual word, 1 means
           that the letter is in the word but placed incorrectly, and 2 means the letter is in the word and in the
           correct position
         - There are five rows, one for each letter (in order) in the guess.

  Examples
  >>> get_feedback_array(_convert_word_to_array('LEVER'), _convert_word_to_array('EATEN'))
      array([[12,  0],
             [ 5,  1],
             [22,  0],
             [ 5,  2],
             [18,  0]])

  >>> get_feedback_array(_convert_word_to_array('MOMMY'), _convert_word_to_array('MADAM'))
      array([[13,  2],
             [15,  0],
             [13,  1],
             [13,  0],
             [25,  0]])

  >>> get_feedback_array(_convert_word_to_array('ARGUE'), _convert_word_to_array('MOTTO'))
      array([[ 1,  0],
             [18,  0],
             [ 7,  0],
             [21,  0],
             [ 5,  0]])

  """
  res = []
  guess_str = _convert_array_to_word(guess_array)
  secret_word_str = _convert_array_to_word(secret_word_array)

  # Get feedback
  feedback_str: str = _get_feedback_str(guess_str, secret_word_str)

  # Convert to an array
  return _convert_feedback_str_to_feedback_array(guess_str, feedback_str)

def get_state_array(guess_arrays: list[np.array], secret_word_array: np.array, remaining_guesses: int, max_guesses: int) -> np.array:
  """
  Convert a list of guesses and feedbacks to a np.array
  This will be the state (representation of the current situtation) for the model

  Args:
    guess_arrays (list[np.array]): A list of numerical representations of guessed words.
    secret_word_array (np.array): A numerical representation of the secret word.
    remaining_guesses (int): Number of guesses remaining
    max_guesses (int): Max guesses allowd

  Returns:
      np.array: A 1D np.array representing the state, of shape (287,) and datatype float-32. It's a concatenation of three binary vectors and one scalar:
         - absent_letters_vector (26 elements): Indicates if a letter is proven absent.
         - required_letter_nums_and_positions_vector (130 elements): Indicates if a letter is present in a specific position.
         - required_letter_nums_and_wrong_positions_vector (130 elements): Indicates if a letter is present but not in a specific position.
         - guesses_remaining (1 element): A normalized guesses remaining (is remaining_guesses divided by max_guesses)

  Source: https://towardsdatascience.com/finding-the-best-wordle-opener-with-machine-learning-ce81331c5759/#:~:text=Word%20Encoding%20Dataset.,in%20%7B0%2C1%7D.
  """
  feedback_arrays: list[np.array] = [] # Construct arrays of data
  for guess_a in guess_arrays:
    feedback_arrays.append(get_feedback_array(guess_a, secret_word_array))

  # Find letters that have been shown to be absent
  absent_letter_nums: list[int] = []
  for feedback_array in feedback_arrays:
    for feedback_array_row in feedback_array:
      if feedback_array_row[1] == 0:
        absent_letter_nums.append(feedback_array_row[0])

  # Find letters that have been shown to be present in particular positions in the actual
  required_letter_nums_and_positions: list[tuple[int, int]] = []
  for feedback_array in feedback_arrays:
    for i, feedback_array_row in enumerate(feedback_array): # The index shows which letter it is (0th, 1st, 2nd, etc)
      if feedback_array_row[1] == 2:
        required_letter_nums_and_positions.append((feedback_array_row[0], i))

  # Find letters that have been shown to present, but not in a particular position in the actual word
  required_letter_nums_and_wrong_positions: list[tuple[int, int]] = []
  for feedback_array in feedback_arrays:
    for i, feedback_array_row in enumerate(feedback_array):
      if feedback_array_row[1] == 1:
        required_letter_nums_and_wrong_positions.append((feedback_array_row[0], i))

  # Convert this data to binary vectors (filled with ones and zeros)
  # Fill this in with 26 zeros
  # Each zero represents if a given letter is proven to be absence
  # Index 0 --> A binary zero / one as to whether or not 'A' is absent
  # Index 1 --> A binary zero / one as to whether or not 'B' is absent
  # ... Repeats for all 26 letters in the alphabet
  absent_letters_vector: list[int] = [0 for i in range(26)]
  for letter_num in absent_letter_nums:
    # letter_num starts at 0, and lists start at zero
    # For example, a letter_num of 1 means 'a', which would be the 0th element in the vector
    absent_letters_vector[letter_num - 1] = 1

  # Fill this in with 130 zeros
  # Each zero represents if a given letter is proven to be in a given position
  # Index 0 --> A binary zero / one as to whether or not 'A' is the 0th letter
  # Index 1 --> A binary zero / one as to whether or not 'A' is the 1th letter
  # Index 2 --> A binary zero / one as to whether or not 'A' is the 2th letter
  # Index 3 --> A binary zero / one as to whether or not 'A' is the 3th letter
  # Index 4 --> A binary zero / one as to whether or not 'A' is the 4th letter
  # ... Repeats for each letter (26) in each possible positon (5), total is 130.
  required_letter_nums_and_positions_vector: list[int] = [0 for i in range(130)]
  for required_letter_num, required_letter_position in required_letter_nums_and_positions:
    index = (required_letter_num - 1) * 5 + required_letter_position
    required_letter_nums_and_positions_vector[index] = 1

  # Fill this in with 130 zeros
  # Each zero represents if a given letter is proven to NOT be given position
  # Index 0 --> A binary zero / one as to whether or not 'A' is NOT 0th letter
  # Index 1 --> A binary zero / one as to whether or not 'A' is NOT the 1th letter
  # Index 2 --> A binary zero / one as to whether or not 'A' is NOT the 2th letter
  # Index 3 --> A binary zero / one as to whether or not 'A' is NOT the 3th letter
  # Index 4 --> A binary zero / one as to whether or not 'A' is NOT the 4th letter
  # ... Repeats for each letter (26) in each possible positon (5), total is 130.
  required_letter_nums_and_wrong_positions_vector: list[int] = [0 for i in range(130)]
  for required_letter_num, wrong_letter_position in required_letter_nums_and_wrong_positions:
    index = (required_letter_num - 1) * 5 + wrong_letter_position
    required_letter_nums_and_wrong_positions_vector[index] = 1

  # Create a normalized guesses-remaining
  normalized_guesses_remaining_scalar = np.array([remaining_guesses / max_guesses], dtype=np.float32)

  return np.concatenate([
      np.array(absent_letters_vector, dtype=np.float32),
      np.array(required_letter_nums_and_positions_vector, dtype=np.float32),
      np.array(required_letter_nums_and_wrong_positions_vector, dtype=np.float32),
      normalized_guesses_remaining_scalar
  ])