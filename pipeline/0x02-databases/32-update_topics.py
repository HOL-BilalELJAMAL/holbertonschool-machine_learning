#!/usr/bin/env python3
"""
32-update_topics.py
Module that defines a function called update_topics
"""


def update_topics(mongo_collection, name, topics):
    """
    Function that changes all topics of a school document based on the name

    Args:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        name [string]:
            the school name to update
        topics [list of strings]:
            list of topics approached in the school
    """
    mongo_collection.update_many({'name': name},
                                 {'$set': {'topics': topics}})
