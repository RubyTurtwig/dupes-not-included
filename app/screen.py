import logging
import operator
import os
import pickle
import time
import json

import mss
import numpy as np
import pyautogui
from PIL import Image

import cv2

if __name__ == "__main__":
    import helpers
else:
    from . import helpers

logger = logging.getLogger(__name__)


class Screen:

    def __init__(self, width=1920, height=1080):
        """ Screen object. """

        DIR = 'assets/'
        RES = '1080p/'

        self.monitor = {'top': 0, 'left': 0, 'width': width, 'height': height}
        self.screen = mss.mss()

        self.shuffle_template = cv2.imread(f'{DIR}shuffle.png', 0)
        self.number_templates = {
            1: cv2.imread(f'{DIR}{RES}+1.png', 0),
            2: cv2.imread(f'{DIR}{RES}+2.png', 0),
            3: cv2.imread(f'{DIR}{RES}+3.png', 0),
            4: cv2.imread(f'{DIR}{RES}+4.png', 0),
            5: cv2.imread(f'{DIR}{RES}+5.png', 0),
            6: cv2.imread(f'{DIR}{RES}+6.png', 0),
            7: cv2.imread(f'{DIR}{RES}+7.png', 0),
            8: cv2.imread(f'{DIR}{RES}+8.png', 0),
            -1: cv2.imread(f'{DIR}{RES}-1.png', 0),
            -2: cv2.imread(f'{DIR}{RES}-2.png', 0),
            -3: cv2.imread(f'{DIR}{RES}-3.png', 0),
        }
        self.attr_templates = {
            'athletics': cv2.imread(f'{DIR}attributes/athletics_positive.png', 0),
            'construction': cv2.imread(f'{DIR}attributes/construction_positive.png', 0),
            'cooking': cv2.imread(f'{DIR}attributes/cooking_positive.png', 0),
            'creativity': cv2.imread(f'{DIR}attributes/creativity_positive.png', 0),
            'digging': cv2.imread(f'{DIR}attributes/digging_positive.png', 0),
            'farming': cv2.imread(f'{DIR}attributes/farming_positive.png', 0),
            'kindness': cv2.imread(f'{DIR}attributes/kindness_positive.png', 0),
            'learning': cv2.imread(f'{DIR}attributes/learning_positive.png', 0),
            'ranching': cv2.imread(f'{DIR}attributes/ranching_positive.png', 0),
            'strength': cv2.imread(f'{DIR}attributes/strength_positive.png', 0),
            'tinkering': cv2.imread(f'{DIR}attributes/tinkering_positive.png', 0)
        }
        self.trait_templates = {
            'anemic': cv2.imread(f'{DIR}traits/anemic.png', 0),
            'biohazardous': cv2.imread(f'{DIR}traits/biohazardous.png', 0),
            'bottomless stomach': cv2.imread(f'{DIR}traits/bottomless_stomach.png', 0),
            'buff': cv2.imread(f'{DIR}traits/buff.png', 0),
            'caregiver': cv2.imread(f'{DIR}traits/caregiver.png', 0),
            "diver's lungs": cv2.imread(f'{DIR}traits/divers_lungs.png', 0),
            'early bird': cv2.imread(f'{DIR}traits/early_bird.png', 0),
            'flatulent': cv2.imread(f'{DIR}traits/flatulent.png', 0),
            'gastrophobia': cv2.imread(f'{DIR}traits/gastrophobia.png', 0),
            'germ resistant': cv2.imread(f'{DIR}traits/germ_resistant.png', 0),
            'gourmet': cv2.imread(f'{DIR}traits/gourmet.png', 0),
            'grease monkey': cv2.imread(f'{DIR}traits/grease_monkey.png', 0),
            'interior decorator': cv2.imread(f'{DIR}traits/interior_decorator.png', 0),
            'iron gut': cv2.imread(f'{DIR}traits/iron_gut.png', 0),
            'irritable bowel': cv2.imread(f'{DIR}traits/irritable_bowel.png', 0),
            'loud sleeper': cv2.imread(f'{DIR}traits/loud_sleeper.png', 0),
            'mole hands': cv2.imread(f'{DIR}traits/mole_hands.png', 0),
            'mouth breather': cv2.imread(f'{DIR}traits/mouth_breather.png', 0),
            'narcoleptic': cv2.imread(f'{DIR}traits/narcoleptic.png', 0),
            'night owl': cv2.imread(f'{DIR}traits/night_owl.png', 0),
            'noodle arms': cv2.imread(f'{DIR}traits/noodle_arms.png', 0),
            'pacifist': cv2.imread(f'{DIR}traits/pacifist.png', 0),
            'quick learner': cv2.imread(f'{DIR}traits/quick_learner.png', 0),
            'simple tastes': cv2.imread(f'{DIR}traits/simple_tastes.png', 0),
            'slow learner': cv2.imread(f'{DIR}traits/slow_learner.png', 0),
            'small bladder': cv2.imread(f'{DIR}traits/small_bladder.png', 0),
            'squeamish': cv2.imread(f'{DIR}traits/squeamish.png', 0),
            'twinkletoes': cv2.imread(f'{DIR}traits/twinkletoes.png', 0),
            'uncultured': cv2.imread(f'{DIR}traits/uncultured.png', 0),
            'yokel': cv2.imread(f'{DIR}traits/yokel.png', 0)
        }
        self.interest_templates = {
            'art': cv2.imread(f'{DIR}interests/art.png', 0),
            'dig': cv2.imread(f'{DIR}interests/dig.png', 0),
            'farm': cv2.imread(f'{DIR}interests/farm.png', 0),
            'operate': cv2.imread(f'{DIR}interests/operate.png', 0),
            'research': cv2.imread(f'{DIR}interests/research.png', 0),
            'tidy': cv2.imread(f'{DIR}interests/tidy.png', 0)
        }
        
        time.sleep(5)  # Time to tab out to ONI.
        
        self.frame = self.take_screenshot()

        self.boxes = None
        self.shuffle_buttons = None

        self.dupe = 0

    def take_screenshot(self):
        """ Return array of pixels in screenshot. """

        return np.asarray(self.screen.grab(self.monitor))

    def take_screenshot_dupe(self, dupe: int):
        """ Take screenshot of duplicant box. """

        box = self.boxes[dupe]

        image = np.asarray(self.screen.grab(box))

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_duplicant_box(self):
        """ Extract three duplicant boxes from frame. """

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        _, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        image = cv2.bitwise_not(image)

        contours, _ = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            bounds = {'top': y, 'left': x, 'width': w, 'height': h}

            # Actual size is (0.90)
            if w / h > 0.89 and w / h < 0.91 and w > 100 and h > 100:
                if bounds not in boxes:
                    boxes.append(bounds)

        boxes.sort(key=lambda x: x['left'])

        logger.info(f'Found boxes: {boxes}.')

        if len(boxes) != 3:
            raise NotImplementedError(f'Must find 3 duplicants!')

        return boxes

    def get_shuffle_buttons(self):
        """ Extract shuffle button from frame. """

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        _, image = cv2.threshold(image, 45, 255, cv2.THRESH_BINARY)
        image = cv2.bitwise_not(image)

        contours, _ = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        shuffle_buttons = []
        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)

            bounds = (x, y)

            # Actual size is (3.75).
            if w / h > 3.5 and w / h < 4 and w > 90 and h > 20:
                if y < 600:  # Avoid being confused with back button.
                    if bounds not in shuffle_buttons:  # Problem with duplicate contours being found.
                        shuffle_buttons.append(bounds)

        shuffle_buttons.sort(key=lambda x: x[0])  # Sory from left to right.

        logger.info(f'Found shuffle buttons: {shuffle_buttons}.')

        if len(shuffle_buttons) != 3:
            raise NotImplementedError(f'Must find 3 duplicants!')

        return shuffle_buttons

    def click_shuffle_button(self, dupe: int = 0):
        """ Click shuffle button for dupe number. """

        x, y = self.shuffle_buttons[dupe]

        pyautogui.click(x=x, y=y)

    def get_attributes(self, image: np.array):
        """ Return dict of attribute: value for attributes from array of dupe box. """

        attributes = {}
        attribute_coords = {}

        for attribute, template in self.attr_templates.items():

            results = cv2.matchTemplate(
                image, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(results)

            x, y = max_loc

            x = round(x/50)  # Introduce room for error.
            y = round(y/15)

            attribute_coords[(x, y)] = attribute

            logger.info(f'Found {attribute} at {(x, y)}.')

        threshold = 0.7
        for number, template in self.number_templates.items():

            matches = cv2.matchTemplate(
                image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(matches >= threshold)
            xs = locations[1]
            ys = locations[0]

            for x, y in zip(xs, ys):

                value = matches[y][x]

                logger.info(f'Found {number} at {(x, y)} with value {value}.')

                x = round(x/50)  # Match error function above.
                y = round(y/15)

                if (x, y) in attribute_coords:

                    attribute = attribute_coords[(x, y)]

                    if attribute not in attributes:

                        attributes[attribute] = (number, value)
                        logger.info(
                            f'Matched {attribute} to {(number, value)}.')

                    else:

                        if attributes[attribute][1] < value:

                            attributes[attribute] = (number, value)
                            logger.info(
                                f'Matched {attribute} to {(number, value)}.')

        return {attribute: value[0] for attribute, value in attributes.items()}

    def get_traits(self, image: np.array):
        """ Get positive and negative traits from image. """

        traits = {'positive': [], 'negative': []}

        for trait, template in self.trait_templates.items():

            matches = self.find_image(image, template, threshold=0.9)

            if len(matches) == 1:

                if trait in {'buff', 'caregiver', "diver's lungs", 'early bird', 'germ resistant', 
                    'gourmet', 'grease monkey', 'interior decorator', 'iron gut', 'mole hands', 
                    'night owl', 'quick learner', 'simple tastes', 'twinkletoes', 'uncultured'}:
                    traits['positive'].append(trait)
                else:
                    traits['negative'].append(trait)

            elif len(matches) >= 1:
                raise NotImplementedError(f'More than one of {trait} found.')

        return traits

    def get_interests(self, image: np.array):

        interests = []

        for interest, template in self.interest_templates.items():

            matches = self.find_image(image, template, threshold=0.9)

            if len(matches) == 1:
                interests.append(interest)

            elif len(matches) >= 1:
                raise NotImplementedError(
                    f'More than one of {interest} found.')

        return interests

    def find_image(self, image: np.array, template: np.array, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
        """ 
        Return x, y of top left corners if image found in template. 
        Return [] if image not found.
        """

        coords = []

        res = cv2.matchTemplate(
            image, template, method)

        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            matches = np.where(res <= threshold)
        else:
            matches = np.where(res >= threshold)

        for i in range(np.shape(matches)[1]):
            coords.append((matches[1][i], matches[0][i]))

        logger.info(f'find_image: {coords} found.')

        return coords

    def create_attributes(self, attributes, traits, interests):
        """ Return attributes of duplicant used for comparison. """

        res = {}
        attribute_list = ('athletics', 'construction', 'cooking', 'creativity', 'digging',
                          'farming', 'kindness', 'learning', 'ranching', 'strength', 'tinkering')

        res['attributes'] = {}

        for attribute in attribute_list:

            if attribute not in attributes:

                res['attributes'][attribute] = 0

            else:

                res['attributes'][attribute] = attributes[attribute]

        res['positive'] = traits['positive']
        res['negative'] = traits['negative']

        res['interests'] = interests

        return res

    def compare_current_duplicant(self, attributes, config: dict):
        """ 
        Return True if current duplicant can be accepted. 
        Config dict must be in the form attributes: {attribute: (operator, number)}.
        Config reads one positive attribute as must, more than one as or. 
        Config reads negative attributes as not.
        Config reads interests as must.
        """

        logger.debug(f'Duplicant attributes found. \n {attributes}')

        for attribute, value in attributes['attributes'].items():

            if attribute in config['attributes']:

                operator, number = config['attributes'][attribute]

                if not operator(value, number):

                    logger.debug(
                        f'Attribute: {attribute} has value: {value}, but operator {operator}, number {number} returned False.')
                    return False

        positive = config['positive']
        negative = config['negative']

        if len(positive) == 1:
            if positive[0] not in attributes['positive']:
                logger.debug(
                    f'Trait {positive[0]} not found in {attributes["positive"]}.')
                return False

        elif len(positive) > 1:
            if not any(trait in attributes['positive'] for trait in positive):
                logger.debug(
                    f'No traits in {positive} found in {attributes["positive"]}')
                return False

        if len(negative) >= 1:
            if any(trait in attributes['negative'] for trait in negative):
                logger.debug(
                    f'Trait in {negative} found in {attributes["negative"]}')
                return False

        for interest in config['interests']:
            if interest not in attributes['interests']:
                return False

        return True

    def load_config(self, name): 
        """ Return duplicant configuration from file name. """ 
        
        if not name.endswith('.json'):  # GUI should return full file path. 

            DIR = 'duplicants/'
            name = os.path.join(DIR, name + '.json')

        operators = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '=':operator.eq}

        with open(name, 'r') as f: 
            config = json.load(f)
        
        for attribute, (op, value) in config['attributes'].items(): 

            config['attributes'][attribute] = (operators[op], value)
        
        return config


    def run(self, dupe, config: str):
        """ Run program with config. Takes in a template name as config. """

        config = self.load_config(config)

        match = False

        self.boxes = self.get_duplicant_box()
        self.shuffle_buttons = self.get_shuffle_buttons()

        dupes = 0
        start = time.time()

        while not match:

            dupe_frame = self.take_screenshot_dupe(dupe)
            attributes = self.create_attributes(
                self.get_attributes(dupe_frame), self.get_traits(dupe_frame), self.get_interests(dupe_frame))
            match = self.compare_current_duplicant(attributes, config)

            if match:
                break

            else:
                self.click_shuffle_button(dupe)
                dupes += 1

        self.dupe += 1  
        end = time.time()

        logger.debug(f'Took {end-start} seconds to run {dupes} duplicants.')


if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    config = {'attributes': {'learning': (operator.gt, 2)}, 'positive': [], 'negative': [
        'slow learner', 'noodle arms', 'loud sleeper', 'narcoleptic'], 'interests': ['research']}

    s = Screen()
    time.sleep(5)
    ss = s.take_screenshot_dupe(0)
    
    # print(s.get_interests(ss))
    # s.run(0, config)
