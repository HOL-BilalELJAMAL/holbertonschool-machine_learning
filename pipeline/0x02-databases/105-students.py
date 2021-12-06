#!/usr/bin/env python3
"""
105-students.py
Module that defines a function called top_students
"""


def top_students(mongo_collection):
    """
    Function that finds list of all schools sorted by average score

    Args:
        mongo_collection [pymongo]: the MongoDB collection to use top must be
        ordered average score must be part of each item returns
        with key=averageScore

    Returns:
        list of all students sorted by average score
    """
    students = []
    documents = mongo_collection.find()
    for student in documents:
        total_score = 0
        topics = student["topics"]
        for project in topics:
            total_score += project["score"]
        average_score = total_score / len(topics)
        student["averageScore"] = average_score
        students.append(student)
    students = sorted(students, key=lambda i: i["averageScore"], reverse=True)
    return students
