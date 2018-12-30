import operator

from app.screen import Screen

screen = Screen()

duplicant_number = 0

# Config file in the following format: 
# {attributes: {attribute: (operator, number)}, 
# positive: [one positive trait to match exactly, more than one trait to match any one], 
# negative: [negative traits to exclude], 
# interests: [interests that duplicant must have]}

config = {'attributes': {'learning': (operator.gt, 2)}, 'positive': [], 'negative': [
    'slow learner', 'noodle arms', 'loud sleeper', 'narcoleptic'], 'interests': ['research']}

screen.run(0, config)
