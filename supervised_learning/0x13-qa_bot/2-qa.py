#!/usr/bin/env python3
"""
2-qa.py
Module that defines a function called answer_loop
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Function that answers questions from a reference text

    Args:
        reference is the reference text
    """
    while True:
        question = input('Q: ')
        question = question.lower()
        if question == 'exit' or question == 'quit' or\
                question == 'goodbye' or question == 'bye':
            print('A: Goodbye')
            break
        else:
            answer = question_answer(question, reference)
            if answer is None:
                answer = 'Sorry, I do not understand your question.'
            print('A: {}'.format(answer))
