#!/usr/bin/env python3
"""
1-loop.py
Module that defines a loop
"""

if __name__ == '__main__':
    """infinite loop"""
    while True:
        question = input('Q: ')
        question = question.lower()
        if question == 'exit' or question == 'quit' or\
                question == 'goodbye' or question == 'bye':
            print('A: Goodbye')
            break
        else:
            print('A: ')
