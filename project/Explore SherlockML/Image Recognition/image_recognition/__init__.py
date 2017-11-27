import sys
import os, inspect

import matplotlib.pyplot as plt
import subprocess as subp


from image_recognition import pretty_asi_plots
from IPython.display import display, Image
from PIL import Image as pil_image

DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 

# hide ticks and box surrounding plots 
pretty_asi_plots.set_parameters([('xtick.major.size', 0),
                               ('ytick.major.size', 0),
                               ('axes.edgecolor', 'white')])
# ASI  styles
pretty_asi_plots.set_asi_styles()

def find_jpgs(directory='.', skip_list=[]):
    '''Find .jpg and .JPG images in the given directory.
    Return the file names of these images.'''
    
    contents = os.listdir(directory)
    
    for skip in skip_list:
        try:
            contents.remove(skip)
        except ValueError:
            pass
    
    return [x for x in contents if x.lower().endswith('.jpg')]

def classify_jpg(jpg, directory='.'):
    '''Classify a given .jpg or .JPG image in given directory.'''
    
    image_file = os.path.join(directory, jpg)
    cmd = 'python \"{}\" --model_dir \"{}\" --image_file \"{}\"'.format(DIR + '/scripts/classify_image/classify_image.py', 
                                                                        DIR + '/model', 
                                                                        image_file)
    
    description = subp.check_output(cmd, shell=True)
    
    return description

def classify_jpgs(directory='.'):
    '''Classify all .jpg and .JPG
    images in given directory.'''
    
    jpgs = find_jpgs(directory)
    
    # if list empty
    if not jpgs:
        raise ValueError("No JPEGS found. Please upload at least one image.")
    
    descriptions = {}
    for jpg in jpgs: 
        descriptions[jpg] = classify_jpg(jpg, directory)
        
    return descriptions

def endlos(jpg, descriptions, directory='.', as_plot=True, max_image_width=500):
    '''Function to be placed in the interactive widget.'''
    
    if as_plot:
        predictions = reformat_results(descriptions[jpg].decode('utf-8'))
        plot(predictions)
    else:
        print(descriptions[jpg])
    
    # get image size
    image_file = os.path.join(directory, jpg)
    image = pil_image.open(image_file)
    image_width, _ = image.size
    
    # constrain width
    if image_width > max_image_width:
        image_width = max_image_width
    
    return Image(image_file, width=image_width)

def reformat_results(results):
    '''Transform string of results 
    returned by model into tuples of
    name as string, score as float'''
    top_5_predictions = results.split('\n')[:5]
    names = [prediction.partition('(')[0].rstrip(' ').capitalize() 
             for prediction in top_5_predictions]
    percent_scores = [float(prediction[-8:-1]) * 100. 
                      for prediction in top_5_predictions]
    return zip(names, percent_scores)

def plot(predictions):
    '''Plot barh chart of predictions'''
    
    # convert to list
    predictions = list(predictions)
    # invert
    predictions = predictions[::-1]
    
    # make names and scores
    names = []
    scores = []
    
    for i in range(0, len(predictions)):
        names.append(predictions[i][0])
        scores.append(predictions[i][1])
    
    fig, ax = plt.subplots()
    n_bars = range(len(names))
    ax.barh(n_bars, scores)
    for i in n_bars:
        ax.annotate(names[i], xy=(scores[i]+1, i),
                    verticalalignment='bottom')
    ax.set_yticks(range(0, len(names)), minor=False)
    ax.set_yticklabels(['{:.1f}%'.format(score) for score in scores], verticalalignment='bottom')
    ax.set_xlim(0, 101)
    ax.set_xticks(range(25, 101, 25))
    ax.xaxis.set_ticklabels(())
    ax.xaxis.grid(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()