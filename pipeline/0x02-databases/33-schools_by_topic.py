#!/usr/bin/env python3
"""
33-schools_by_topic.py
Module that defines a function called schools_by_topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Function that finds list of all schools with a specific topic

    Args:
        mongo_collection [pymongo]:
            the MongoDB collection to use
        topic [string]:
            the topic to search for

    Returns:
        list of schools with the given topic
    """
    schools = []
    documents = mongo_collection.find({'topics': {'$all': [topic]}})
    for doc in documents:
        schools.append(doc)
    return schools
