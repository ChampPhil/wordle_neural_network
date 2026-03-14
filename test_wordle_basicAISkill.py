import unittest
from wordle_secret_words import get_secret_words
from valid_wordle_guesses import get_valid_wordle_guesses
import random
import time

from wordle import get_AI_guess
from wordle import get_feedback

# Force a timeout message (failed test) if test runtime exceeds this number of seconds.
MAX_TIMEOUT_SECONDS = 60 * 60 * 60

def play_wordle(secret_word, secret_words, valid_guesses, get_AI_guess):
    guesses = []
    feedback = []
    game_over = False

    while not game_over:
        guesses.append(get_AI_guess(guesses, feedback, secret_words, valid_guesses))
        feedback.append(get_feedback(guesses[len(guesses)-1], secret_word))
        if len(guesses) >= 6 or guesses[len(guesses)-1] == secret_word:
            game_over=True

    return guesses

class TestEx1(unittest.TestCase):
  def test_basic_requirements(self):
      """get_AI_guess - Basic Requirements"""
      guesses =()
      feedback = ()
      guess = get_AI_guess(guesses, feedback, get_secret_words(), get_valid_wordle_guesses())
      self.assertEqual(True, isinstance(guess, str), "Returned value isn't a string.")
      self.assertEqual(len(guess), 5)
      self.assertEqual(guess in get_valid_wordle_guesses(), True, "Returned word is not a valid Wordle guess.")
  

  def test_wordle_secret_words(self, set_leaderboard_value=None):
      """get_AI_guess - Basic Profiler (shuffled wordlist w/ ~2,300 5-letter words)"""
      print("\n------------------------Basic AI Profiler: All Wordle secret words----------------------------")
      wins = 0
      skill = 0
      secret_word_list = list(get_secret_words())
      numWords = len(secret_word_list)
      random.shuffle(secret_word_list)
      start_time = time.time()

      for secret_word in secret_word_list:
          self.assertEqual(len(get_secret_words()), numWords, "You should not modify underlying word list")
          guesses = play_wordle(secret_word, get_secret_words(), get_valid_wordle_guesses(), get_AI_guess)
          self.assertLessEqual(len(guesses), 6, "You should not modify guesses within get_AI_guess.")
          for guess in guesses:
            self.assertEqual(True, isinstance(guess, str), "Returned guesses should be strings.")

          if secret_word==guesses[len(guesses)-1]:
              wins += 1
              skill += len(guesses)
          
          if time.time() - start_time > MAX_TIMEOUT_SECONDS:
              print(f"The test timed out after {MAX_TIMEOUT_SECONDS / 60} minutes. See if you can improve the efficiency of the function.")
              self.assertEqual(True, False)
              return
      elapsed_time = time.time() - start_time

      accuracy = (wins/numWords)*100
      print(f"Accuracy: {wins}/{numWords} games - {accuracy}%")
      print(f"Time taken: {elapsed_time}sec")
      skill_out = "N/A"
      if wins > 0:
        print(f"Skill (average guesses per win): {skill/wins}%")
        skill_out = skill/wins
      else:
        print(f"Skill (average guesses per win): N/A - NO GAMES WON")

      self.assertEqual(True, True)
      
  def test_wordle_hard_words(self, set_leaderboard_value=None):
      """get_AI_guess - Hardest Words for Humans (per NY Times)"""
      print("\n------------------------Basic AI Profiler: 10 Hardest Human Words----------------------------")

      wins = 0
      skill = 0
      hardest_words = ["PARER", "ATONE", "COYLY", "JOKER", "JAZZY", "CATCH", "KAZOO", "NANNY", "MUMMY", "JUDGE"]
      random.shuffle(hardest_words)
      numWords = len(hardest_words)
      start_time = time.time()

      for secret_word in hardest_words:
          guesses = play_wordle(secret_word, get_secret_words(), get_valid_wordle_guesses(), get_AI_guess)
          self.assertLessEqual(len(guesses), 6, "You should not modify guesses within get_AI_guess.")
          for guess in guesses:
            self.assertEqual(True, isinstance(guess, str), "Returned guesses should be strings.")
            self.assertEqual(guess in get_valid_wordle_guesses(), True, "Returned guess is not a valid Wordle guess.")

          if secret_word==guesses[len(guesses)-1]:
              wins += 1
              skill += len(guesses)
              self.assertTrue(time.time() - start_time < MAX_TIMEOUT_SECONDS, f"The test timed out after {MAX_TIMEOUT_SECONDS / 60} minutes. See if you can improve the efficiency of get_AI_guess.")
              
      accuracy = (wins/numWords) * 100
      print(f"Hardest Words for Humans Accuracy: {wins}/{numWords} -{accuracy}%")
 
      if wins > 0:
        print(f"Skill (average guesses per win): {skill/wins}")
      else:
        print(f"Skill (average guesses per win): N/A - NO GAMES WON")


  def test_random_words(self):
      """get_AI_guess - Advanced Profiler (shuffled wordlist w/ randomly chosen words from a list of ~13,000 5-letter words)"""
      print("\n------------------------Advanced AI Profiler: Random words from valid wordle guesses---------")
      wins = 0
      skill = 0
      numWords = len(get_secret_words())

      random_word_list = list(get_valid_wordle_guesses())
      random.shuffle(random_word_list)
      random_word_list = random_word_list[0:numWords]

      start_time = time.time()

      for secret_word in random_word_list:
          self.assertEqual(len(get_secret_words()), numWords, "You should not modify underlying word list")
          guesses = play_wordle(secret_word, get_secret_words(), get_valid_wordle_guesses(), get_AI_guess)
          self.assertLessEqual(len(guesses), 6, "You should not modify guesses within get_AI_guess.")
          for guess in guesses:
            self.assertEqual(True, isinstance(guess, str), "Returned guesses should be strings.")

          if secret_word==guesses[len(guesses)-1]:
              wins += 1
              skill += len(guesses)
          
          if time.time() - start_time > MAX_TIMEOUT_SECONDS:
              print(f"The test timed out after {MAX_TIMEOUT_SECONDS / 60} minutes. See if you can improve the efficiency of the function.")
              self.assertEqual(True, False)
              return
      elapsed_time = time.time() - start_time
          
      print(f"Accuracy: {wins}/{numWords} - {(wins/numWords)*100}%")
      print(f"Time taken: {elapsed_time}sec")
      if wins > 0:
        print(f"Skill (average guesses per win): {skill/wins}")
      else:
        print(f"Skill (average guesses per win): N/A - NO GAMES WON")

if __name__ == '__main__':
   	#unittest.main()
    unittest.main(argv=[''], defaultTest='TestEx1.test_wordle_secret_words')