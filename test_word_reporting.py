import pandas as pd
import word_reporting

def test_get_word_set():
    # Test 1: Test that the function returns the correct output for a given input dataframe.
    df = pd.DataFrame({'text': ['This is a sample sentence.', 'Another sample sentence.']})
    assert word_reporting.get_word_set(df, text_col='text') == ['Another', 'This', 'a', 'is', 'sample', 'sentence.']
    print('test 1 get_word_set passed')

    # Test 2: Test that the function returns an empty list when the input dataframe is empty.
    df = pd.DataFrame({'text': []})
    assert word_reporting.get_word_set(df, text_col='text') == []
    print('test 2 get_word_set passed')

    # Test 3: Test that the function returns the correct output when the column containing text contains special characters and numbers.
    df = pd.DataFrame({'text': ['This is a sample sentence with numbers 123 and special characters !@#$.']})
    assert word_reporting.get_word_set(df, text_col='text') == ['!@#$.', '123', 'This', 'a', 'and', 'characters', 'is', 'numbers', 'sample', 'sentence', 'special', 'with']
    print('test 3 get_word_set passed')

    # Test 4: Test that the function handles duplicate words correctly.
    df = pd.DataFrame({'text': ['This is a sample sentence.', 'Another sample sentence.', 'This is a duplicate sentence.']})
    assert word_reporting.get_word_set(df, text_col='text') == ['Another', 'This', 'a', 'duplicate', 'is', 'sample', 'sentence.']
    print('test 4 get_word_set passed')

    # Test 5: Test that the function returns the correct output for a dataframe with multiple rows.
    df = pd.DataFrame({'text': ['This is a sample sentence.', 'Another sample sentence.', 'This is a duplicate sentence.'], 
                       'other_text': ['This is some other text.', 'More text here.', 'Duplicate text here.']})
    assert word_reporting.get_word_set(df, text_col='text') == ['Another', 'This', 'a', 'duplicate', 'is', 'sample', 'sentence.']
    print('test 5 get_word_set passed')

def main():
    test_get_word_set()

if __name__ == '__main__':
    main()

