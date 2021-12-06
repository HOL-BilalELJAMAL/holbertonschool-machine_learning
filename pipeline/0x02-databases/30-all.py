#!/usr/bin/env python3
"""
30-all.py
Module that defines a function called list_all
"""


def list_all(mongo_collection):
    """
    Function that lists all documents in given MongoDB collection

    Args:
        mongo_collection: the collection to use

    Returns:
        list of all documents or 0 if no documents found
    """
    all_docs = []
    collection = mongo_collection.find()
    for document in collection:
        all_docs.append(document)
    return all_docs
