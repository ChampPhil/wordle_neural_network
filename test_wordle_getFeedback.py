import unittest
from wordle import get_feedback

class TestEx1(unittest.TestCase):

  def test_1(self):
      """get_feedback - AMBER presentation example"""
      secret_word = "AMBER"
      guess = "BRAKE"
      expected = "bra-e"

      actual = get_feedback(guess, secret_word)
      self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_2(self):
      """get_feedback - MOTTO presentation example"""
      secret_word = "MOTTO"
      guess = "TOOTH"
      expected = "tOoT-"

      actual = get_feedback(guess, secret_word)
      self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_3(self):
      """get_feedback - Documentation example #1"""
      secret_word = "EATEN"
      guess = "LEVER"
      expected = "-e-E-"  

      actual = get_feedback(guess, secret_word)
      self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_4(self):
       """get_feedback - Documentation example #2"""
       secret_word = "LOWER"
       guess = "LEVER"
       expected = "L--ER"  

       actual = get_feedback(guess, secret_word)
       self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_5(self):
      """get_feedback - Documentation example #3"""
      secret_word = "MADAM"
      guess = "MOMMY"
      expected = "M-m--" 

      actual = get_feedback(guess, secret_word)
      self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_6(self):
      """get_feedback - Documentation example #4"""
      secret_word = "ARGUE"
      guess = "MOTTO"
      expected = "-----"  

      actual = get_feedback(guess, secret_word)
      self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_7(self):
        """get_feedback - Guess is secret word"""
        secret_word = "MOTTO"
        guess = "MOTTO"
        expected = "MOTTO" 

        actual = get_feedback(guess, secret_word)
        self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_8(self):
        """get_feedback - Guess is lowercase"""
        secret_word = "MADAM"
        guess = "mommy"
        expected = "M-m--" 

        actual = get_feedback(guess, secret_word)
        self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_9(self):
        """get_feedback - Guess and secret word are anagrams with shared letter positions"""
        secret_word = "CRATE"
        guess = "TRACE"
        expected = "tRAcE"

        actual = get_feedback(guess, secret_word)
        self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")

  def test_10(self):
        """get_feedback - Guess and secret word are anagrams with no shared letter positions"""
        secret_word = "ANGLE"
        guess = "GLEAN"
        expected = "glean"

        actual = get_feedback(guess, secret_word)
        self.assertEqual(actual, expected, f"Secret Word: {secret_word} Guess: {guess} You returned {actual}, but you should have returned {expected}")
  
  def test_11(self):
        """get_feedback - 3Blue1Brown Mistake Examples"""
        secret_words = ["ABIDE", "ERASE", "STEAL", "CREPE"]
        guess = "SPEED"
        expected = ["--e-d", "s-ee-", "S-E--", "-pEe-"]

        for i in range(len(secret_words)):
            actual = get_feedback(guess, secret_words[i])
            self.assertEqual(actual, expected[i], f"Secret Word: {secret_words[i]} Guess: {guess} You returned {actual}, but you should have returned {expected[i]}")

if __name__ == '__main__':
   	unittest.main()