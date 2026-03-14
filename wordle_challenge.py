from wordle_secret_words import get_secret_words
from valid_wordle_guesses import get_valid_wordle_guesses

def starting_two_words(secret_words: set[str], valid_guesses: set[str]) -> set[tuple[str]]:
    '''Determines the best starting two word combination(s) from valid_guesses which would eliminate the
       most words from secret_words assuming both guesses return all grey feedback

       Args:
         secret_words (set[str]): A set of potential secret Wordle words
         valid_guesses (set[str]): A set of allowable Wordle guesses

        Returns:
         set[tuple[str]]: A set of tulples of strings representing all the two word combinations 
                          that would eliminate the most words from secret_words
    '''
    ### BEGIN SOLUTION

    ### END SOLUTION 

def most_green_matches(secret_words: set[str], valid_guesses: set[str]) -> set[str]:
    '''Determines the word(s) from valid_guesses make at least one Green match with 
       the most number of words in secret_words 

       Args:
         secret_words (set[str]): A set of potential secret Wordle words
         valid_guesses (set[str]): A set of allowable Wordle guesses

        Returns:
         set[str]: A set of strings containing all the words from valid_guesses which 
                   make at least one Green match with the most number of words in secret_words 
    '''
   ### BEGIN SOLUTION

   ### END SOLUTION 

def most_yellow_matches(secret_words: set[str], valid_guesses: set[str]) -> set[[str]]:
   '''Determines the word(s) from valid_guesses make at least two Yellow matches
      with the most number of words in secret_words

       Args:
         secret_words (set[str]): A set of potential secret Wordle words
         valid_guesses (set[str]): A set of allowable Wordle guesses

        Returns:
         set[str]: A set of strings containing all the words from valid_guesses which 
                   make at least two Yellow matches with the most number of words in secret_words 
    '''
   ### BEGIN SOLUTION

   ### END SOLUTION 

if __name__ == "__main__":
    # TODO: Write your own code to call your functions here
    pass