# dupes-not-included
Shuffle duplicants until it finds one that meets a specified criteria!

1/1/19: I added simple GUI control. 

# How to Use:
1. Download python 3.  
2. Create a .json file with the configuration of the duplicant you want to find, and save it in the duplicants folder. The json file must be in the format shown below.
3. Open the ONI duplicant selection screen. 
4. Run main.py with python.
5. Select the json file created in 2, and then click the duplicant number to shuffle. 
6. Click run, and immediately tab out to ONI. The program should start shuffling in a few seconds. 

# JSON Format
The json file must be in the following format: 
{"attributes": {attribute: \[operator, value]}, 
 "positive": \[positive traits], 
 "negative: \[negative traits], 
 "interests": \[interests]
}

1. The operator can be >, >=, < or <=. 
2. If one positive trait is given, the script will match exactly that trait. If more than one is given, the script will match any one of the given traits. 
3. The script will exclude duplicants with any of the negative traits given. 
4. The script will find duplicants with all of the interests given. 

The example.json file given searches for a duplicant with learning greater than 2, isn't a slow learner, noodle arms, loud sleeper or narcoleptic, and has research as an interest. 

# Notes
I'm just a beginner at this so this script probably contains loads of bugs, doesn't work for any other resolution than 1080p and breaks a whole load of programming practices. Also, it's pretty slow at shuffling duplicants, so try not to put in too specific a criteria. 

There's plenty left to do with this project as well: 
1. Extend its functionality to other resolutions. 
2. Optimizing the script so that it can shuffle through duplicants at a faster rate. 
3. Add ability to parse more than one duplicant at once. 
4. Add ability to build a comparison dictionary. 
5. Add duplicant templates. 
