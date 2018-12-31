import operator

from app.screen import Screen

screen = Screen()

duplicant_number = 0

# Config file in the following format: 
# {attributes: {attribute: (operator, number)}, 
# positive: [one positive trait to match exactly, more than one trait to match any one], 
# negative: [negative traits to exclude], 
# interests: [interests that duplicant must have]}

screen.run(0, 'example')
