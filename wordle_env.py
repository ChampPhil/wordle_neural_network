import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import os
import string
from collections import Counter
from typing import Optional, Callable
from wordle_utils import get_state_array, get_feedback_array, NUMS_TO_CHARACTERS_DICT, CHARACTERS_TO_NUMS_DICT

class WordleEnv(gym.Env):
  """
  Source:
     - Creating an Environment: https://gymnasium.farama.org/introduction/create_custom_env/#environment-init
     - Existing Wordle Neural Network Implementation: https://andrewkho.github.io/wordle-solver/
  """
  def __init__(
      self,
      secret_word_arrays: list[np.array],
      max_guesses: int = 6
    ):
    """
    Initialize the Wordle environment.
    Args:
      secret_word_arrays: Numerical arrays of every secret word (one will be randomly picked at initialization)
      max_guesses: The maximum amount of guesses the person is allowed.
    """
    # Define word banks
    self.secret_word_arrays = secret_word_arrays

    # Define guess tracking (state)
    self.max_guesses = max_guesses
    self.num_guesses_remaining = max_guesses
    self.num_guesses_made = 0
    self.is_last_guess = False

    # Define board tracking (state)
    self.guesses: list[np.array] = []
    self.feedbacks: list[np.array] = []
    self.secret_word: np.array = np.array([])

    # Define what the agent can observe (the input)
    self.observation_space = gym.spaces.Box(
        low=0,
        high=1,
        shape=(288,),
        dtype=np.float32
    )

    # Define all potential actions (the output as a scalar)
    # 0 --> 'abase'
    # 1 ---> 'adorn'
    # ...(up to 2313)
    self.action_space = gym.spaces.Discrete(len(secret_word_arrays))

    # Map action numbers to actual word arrays (and vice versa)
    self._action_to_word_array: dict[int, np.array] = {i: guess_array for i, guess_array in enumerate(secret_word_arrays)}
    # Use tobytes to convert np.ndarray to raw bytes (so it can be a key in a Python dict, meaning it must be unhashable)
    self._word_array_bytes_to_action: dict[bytes, int] = {word_array.tobytes(): i for i, word_array in enumerate(self.secret_word_arrays)}

  def _get_obs(self) -> np.array:
    """
    Convert internal state to observation format - a np.array of sape (286, ).

    Returns:
        dict: Observation with agent and target positions
    """
    # Mapping of the board
    state_array = get_state_array(self.guesses, self.secret_word, self.num_guesses_remaining, self.max_guesses)

    # Add number of valid words left
    n_valid = len(self._filter_valid_words())
    valid_count_feature = np.array([n_valid / len(self.secret_word_arrays)], dtype=np.float32)

    return np.concatenate([
        state_array,
        valid_count_feature
    ])


  def _get_info(self) -> dict[str, int]:
    """
    Creates a dictionary of useful metadata
      - 'num_guesses': The number of guesses made (int)
      - 'num_guesses_remaining': The number of guesses remaining (int)
      - 'is_last_guess': Whether or not it is the last guess (bool)
      - 'num_absent_letters': The number of unique letters proven to be absent from the secret word (int)
      - 'num_correct_positions_found': The number of unique letter-position pairs that are correct (int)
      - 'num_wrong_positions_found': The number of unique letter-position pairs where the letter is present but in the wrong spot (int)
    """
    progress_metrics = self._calculate_progress()
    return {
        'num_guesses': self.num_guesses_made,
        'num_guesses_remaining': self.num_guesses_remaining,
        'is_last_guess': self.is_last_guess,
        'num_absent_letters': progress_metrics['num_absent_letters'],
        'num_correct_positions_found': progress_metrics['num_correct_positions_found'],
        'num_wrong_positions_found': progress_metrics['num_wrong_positions_found'],
    }

  def _calculate_progress(self):
    """
    Calculates metrics indicating the progress of a Wordle game.
    Returns:
        dict: A dictionary containing summary metrics:
            - 'num_absent_letters' (int): The number of unique letters proven to be absent from the secret word.
            - 'num_correct_positions_found' (int): The number of unique letter-position pairs that are correct.
            - 'num_wrong_positions_found' (int): The number of unique letter-position pairs where the letter is present but in the wrong spot.
            - 'num_valid_actions' (int): The total number of valid actions (words) that can be taken.
    """
    # Find letters that have been shown to be absent (UNIQUE)
    absent_letter_nums = set()
    for feedback_array in self.feedbacks:
      for feedback_array_row in feedback_array:
        if feedback_array_row[1] == 0:
          absent_letter_nums.add(feedback_array_row[0])

    # Find letters that have been shown to be present in particular positions in the actual
    required_letter_nums_and_positions: list[tuple[int, int]] = []
    for feedback_array in self.feedbacks:
      for i, feedback_array_row in enumerate(feedback_array): # The index shows which letter it is (0th, 1st, 2nd, etc)
        if feedback_array_row[1] == 2 and (feedback_array_row[0], i) not in required_letter_nums_and_positions:
          required_letter_nums_and_positions.append((feedback_array_row[0], i))

    # Find letters that have been shown to present, but not in a particular position in the actual word
    required_letter_nums_and_wrong_positions: list[tuple[int, int]] = []
    for feedback_array in self.feedbacks:
      for i, feedback_array_row in enumerate(feedback_array):
        if feedback_array_row[1] == 1 and (feedback_array_row[0], i) not in required_letter_nums_and_wrong_positions:
          required_letter_nums_and_wrong_positions.append((feedback_array_row[0], i))

    # Get valid actions
    num_valid_actions = len(self._filter_valid_words())

    # Calcutate summary metrics
    num_absent_letters = len(absent_letter_nums)
    num_correct_positions_found = len(required_letter_nums_and_positions)
    num_wrong_positions_found = len(required_letter_nums_and_wrong_positions)

    return {
        'num_absent_letters': num_absent_letters,
        'num_correct_positions_found': num_correct_positions_found,
        'num_wrong_positions_found': num_wrong_positions_found,
        'num_valid_actions': num_valid_actions
    }

  def _has_won(self, guess: np.array) -> bool:
    """
    Checks if the guessed word matches the secret word.

    Args:
        guess (np.array): The numerical representation of the guessed word.

    Returns:
        bool: True if the guess matches the secret word, False otherwise.
    """
    return np.array_equal(guess, self.secret_word)

  def _filter_valid_words(self) -> list[np.array]:
    """
    Filters the full word list down to only words consistent with all feedback received so far.

    Uses self.guesses, self.feedbacks, and self.secret_word_arrays to determine
    which words are still valid candidates. Applies four constraint types derived from Wordle feedback:
      1. GREEN (correct position): Word must have that letter at that exact position.
      2. YELLOW (wrong position): Word must contain that letter, but NOT at that position.
      3. GREY (absent): Word must NOT contain that letter anywhere (unless it also appeared as green/yellow).
      4. Minimum counts: Word must contain each required letter at least a certain number of times.
      5. Already-guessed words are excluded.

    Returns:
        list[np.array]: A list of word arrays that are consistent with all known feedback.
    """
    # If no guesses yet, all words are valid
    if not self.feedbacks:
        return list(self.secret_word_arrays)

    # Build constraint sets from feedback
    must_have_letters = {}             # (letter_num: min_count) pairs the word MUST contain
    must_have_positions = []           # (letter_num, position) pairs the word MUST match (from green)
    incorrect_letters = set()          # Letters the word CANNOT have anywhere (from grey)
    incorrect_positions = []           # (letter_num, position) pairs where letter CANNOT be (from yellow)

    for feedback_array in self.feedbacks:
        current_guess_confirmed_counts = {i: 0 for i in NUMS_TO_CHARACTERS_DICT.keys()} # Create a {letter-num: min-count} for this feedback only
        for i, row in enumerate(feedback_array):
            letter_num, status = row[0], row[1]
            if status == 2:  # GREEN
                must_have_positions.append((letter_num, i))
                current_guess_confirmed_counts[letter_num] += 1
            elif status == 1: # YELLOW
                incorrect_positions.append((letter_num, i))
                current_guess_confirmed_counts[letter_num] += 1

        # Update overall min_required_letter_counts from this feedback array
        for letter_num, count in current_guess_confirmed_counts.items():
            must_have_letters[letter_num] = max(must_have_letters.get(letter_num, 0), count)


    # Identify truly absent letters (grey, and not confirmed present elsewhere)
    for feedback_array in self.feedbacks:
        for i, row in enumerate(feedback_array):
            letter_num, status = row[0], row[1]
            if status == 0 and must_have_letters.get(letter_num, 0) == 0: # Only add to incorrect if not required at all
                incorrect_letters.add(letter_num)

    # Filter words
    valid_words = []

    for word_array in self.secret_word_arrays:
        # Skip already-guessed words
        if any(np.array_equal(word_array, g) for g in self.guesses):
            continue

        valid = True

        # Check grey: word must NOT contain any fully absent letter
        for letter_num in incorrect_letters:
            if letter_num in word_array:
                valid = False
                break
        if not valid:
            continue

        # Check green: word must have the correct letter at each green position
        for letter_num, pos in must_have_positions:
            if word_array[pos] != letter_num:
                valid = False
                break
        if not valid:
            continue

        # Check minimum letter counts (from green/yellow feedback)
        candidate_word_counts = Counter(word_array)
        for letter_num, min_count in must_have_letters.items():
            if candidate_word_counts[letter_num] < min_count:
                valid = False
                break
        if not valid:
            continue

        # Check yellow (position exclusion): letter cannot be at that specific position
        for letter_num, pos in incorrect_positions:
            if word_array[pos] == letter_num:
                valid = False
                break
        if not valid:
            continue

        valid_words.append(word_array)

    # Safety fallback: if everything got filtered out, return all words
    if not valid_words:
        return list(self.secret_word_arrays)

    return valid_words

  def valid_action_mask(self) -> np.ndarray:
    """
    Returns a boolean mask over the action space indicating which actions are valid.

    Uses _filter_valid_words() to determine which words are consistent with all feedback
    received so far, then maps those words back to their action indices using a precomputed
    lookup table. Actions corresponding to invalid words are set to False so that MaskablePPO
    will assign them zero probability before sampling.

    Returns:
        np.ndarray: A boolean array of shape (n_actions,) where True means the action
                    is valid (word is consistent with feedback) and False means it should
                    be masked out.
    """
    valid_words = self._filter_valid_words() # Get all valid words
    mask = np.zeros(len(self.secret_word_arrays), dtype=bool) # Create mask (for now, all values are zero)

    for word_array in valid_words: # Change valid words (make the mask not convert them to zero)
        action_idx = self._word_array_bytes_to_action[word_array.tobytes()]
        mask[action_idx] = True

    if not np.any(mask): # Safety check if mask zeros everything
        return np.ones(len(self.secret_word_arrays), dtype=bool)

    return mask

  def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    """
    Start a new episode.

    Args:
        seed: Random seed for reproducible episodes
        options: Additional configuration. 
            - If 'secret_word' is a key (as an np.ndarray), will use that vector as the secret word.
            - If 'guesses' is a key (as a list of np.nddarry), will set 'self.guesses' to that
            - If 'feedbacks' is a key (as a list of np.nddarry), will set 'self.feedbacks' to that

    Returns:
        tuple: (observation, info) for the initial state
    """
    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)

    # Get a random secret word OR use the one in options
    if options and 'secret_word' in options and isinstance(options['secret_word'], np.ndarray):
      self.secret_word = options['secret_word']
    else:
      secret_word_index = self.np_random.integers(0, len(self.secret_word_arrays), size=1, dtype=int)[0]
      self.secret_word = self.secret_word_arrays[secret_word_index]

    # Reset other attributes
    self.num_guesses_remaining = self.max_guesses
    self.num_guesses_made = 0
    self.is_last_guess = False
    self.guesses: list[np.array] = options['guesses'] if options and 'guesses' and isinstance(options['guesses'], list) else []
    self.feedbacks: list[np.array] = options['feedbacks'] if options and 'feedbacks' and isinstance(options['feedbacks'], list) else []

    # Get initial observation and info
    observation = self._get_obs()
    info = self._get_info()
    return observation, info

  def step(self, action: int) -> tuple:
    """
    Execute one action within the environment.

    Args:
        action: The action to take (corresponds to a guess tensor)

    Returns:
        tuple: (observation, reward, terminated, truncated, info)
    """

    # Get progress (before taking new action into account)
    old_progress = self._calculate_progress()

    # Map the discrete action (int) to a word array
    guess: np.array = self._action_to_word_array[action]
    feedback: np.array = get_feedback_array(guess, self.secret_word)

    # Update guess tracking
    self.num_guesses_remaining -= 1
    self.num_guesses_made += 1
    self.is_last_guess = self.num_guesses_remaining == 0

    # Update board tracking
    self.guesses.append(guess)
    self.feedbacks.append(feedback)

    # Check if agent has won / or has made its' last guess
    terminated = self._has_won(guess) or self.is_last_guess # Whether the agent reaches a terminal state in the episode
    truncated = False # Whether some other conditions requires the episode to end (out of the ordinary, like a time limit)

    # Calculate reward / progress
    new_progress = self._calculate_progress()
    new_absent_letters = new_progress['num_absent_letters'] - old_progress['num_absent_letters']
    new_correct_positions = new_progress['num_correct_positions_found'] - old_progress['num_correct_positions_found']
    new_wrong_positions = new_progress['num_wrong_positions_found'] - old_progress['num_wrong_positions_found']

    #if self._has_won(guess):
    #  reward = 10 + (6 - self.num_guesses_made) * 2
    #elif self.is_last_guess:
    #  reward = -10
    #else:
    #  reward = -1 + (new_absent_letters + new_correct_positions + new_wrong_positions) / 2

    if self._has_won(guess):
      # Ranges from 22 -- 10
      reward = 10 + (6 - self.num_guesses_made) * 2
    elif self.is_last_guess:
      reward = -10
    else:
      # Reward based on how much this guess narrowed the search space
      old_valid = old_progress['num_valid_actions']  # count before this guess
      new_valid = len(self._filter_valid_words())  # count after
      reduction_ratio = 1 - (new_valid / old_valid) if old_valid > 0 else 0
      reward = reduction_ratio * 2 - 0.5  # ranges from -0.5 to 1.5

    # Return observation, reward, etc
    observation = self._get_obs()
    info = self._get_info()
    return observation, reward, terminated, truncated, info

  def render(self):
    pass