#!/usr/bin/env python3
"""
0-passengers.py
Module that defines a function called availableShips
"""

import requests


def availableShips(passengerCount):
    """
    Function that uses the Star Wars API to return the list of ships that
    can hold passengerCount number of passengers

    Args:
        passengerCount [int]: the number of passenger the ship must be able to
        carry

    Returns:
        [list]: all ships that can hold that many passengers
    """
    if type(passengerCount) is not int:
        raise TypeError(
            "passengerCount must be a positive number of passengers")
    if passengerCount < 0:
        raise ValueError(
            "passengerCount must be a positive number of passengers")
    url = "https://swapi-api.hbtn.io/api/starships/?format=json"
    ships = []
    while url:
        results = requests.get(url).json()
        ships += results.get('results')
        url = results.get('next')
    shipsList = []
    for ship in ships:
        passengers = ship.get('passengers').replace(",", "")
        if passengers != "n/a" and passengers != "unknown":
            if int(passengers) >= passengerCount:
                shipsList.append(ship.get('name'))
    return shipsList
