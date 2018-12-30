import pickle
import logging

from app import screen as s


def test_positive_attrs():

    screen: s.Screen = s.Screen()

    with open('tests/dupe.pkl', 'rb') as f:
        img = pickle.load(f)

    attrs = screen.get_positive_attributes(img)

    assert attrs == {'tinkering': 5, 'strength': 3}

    with open('tests/dupe2.pkl', 'rb') as f: 
        img = pickle.load(f)

    attrs = screen.get_positive_attributes(img)

    assert attrs == {'kindness': 5, 'ranching': 4, 'creativity': 3}

    
