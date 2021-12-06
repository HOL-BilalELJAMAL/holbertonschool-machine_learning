#!/usr/bin/env python3
"""
31-insert_school.py
Module that defines a function called insert_school
"""


def insert_school(mongo_collection, **kwargs):
    """
    Function that inserts a new document in a MongoDB collection

    Args:
        kwargs: the new document to add

    Retruns:
        the new _id
    """
    document = mongo_collection.insert_one(kwargs)
    return document.inserted_id
