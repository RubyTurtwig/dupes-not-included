import pickle
import logging

from app.screen import Screen


with open('tests/test-dupe-1.pkl', 'rb') as f:
    image = pickle.load(f)

s = Screen()


def test_attributes():
    """ Test if screen can read attributes properly. """

    attributes = s.get_attributes(image)

    correct_attributes = {'athletics': 1,
                          'digging': 3, 'creativity': 8, 'farming': 2}

    assert attributes == correct_attributes


def test_traits():
    """ Test if screen can read traits properly. """

    traits = s.get_traits(image)

    correct_traits = {'positive': ['interior decorator'], 'negative': ['squeamish']}

    assert traits == correct_traits


def test_interests(): 
    """ Test if screen can read interests properly. """

    interests = s.get_interests(image)

    correct_interests = ['dig']

    assert interests == correct_interests