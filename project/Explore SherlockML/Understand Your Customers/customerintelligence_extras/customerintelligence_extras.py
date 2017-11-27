import folium
from folium import plugins
import inspect
import os
import matplotlib as mpl
import matplotlib.pyplot as pyplot

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py


from sherlockml import census
from sherlockml import geoplot
from sherlockml import opendata
from sherlockml.model.unsuperviseduplifttree import analyse_feature_split

DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + '/'


def display_colorbar(colormap_name, title, ticks=[0, 1],
                     labels=['Low', 'High'], fontsize=16):
    """ Display a matplotlib colormap. """

    fig, axes = pyplot.subplots(figsize=(10, 0.5))

    # Normalise colours to match the range of values
    norm = mpl.colors.Normalize(vmin=min(ticks), vmax=max(ticks))

    # Plot colourbar
    cbr = mpl.colorbar.ColorbarBase(
        axes,
        cmap=mpl.cm.get_cmap(colormap_name),
        norm=norm,
        orientation='horizontal',
        ticks=ticks
    )
    axes.set_title(title, fontsize=fontsize)
    axes.set_xticklabels(labels, fontsize=fontsize)

    pyplot.show()

def available_metrics():
    return ['Persons_per_Hectare',
                         'Persons_per_Household',
                         'Fraction_Households_owned_by_tenants',
                         'Fraction_Households_socially_rented',
                         'Fraction_Households_privately_rented',
                         'Fraction_of_persons_under_20',
                         'Fraction_of_persons_20-29',
                         'Fraction_of_persons_30-44',
                         'Fraction_of_persons_45-64',
                         'Fraction_of_persons_65_and_over',
                         'Fraction_white',
                         'Fraction_UK_born',
                         'Fraction_adult_population_single',
                         'Fraction_married_partnership_widowed',
                         'Fraction_divorced_or_separated',
                         'Fraction_unemployed',
                         'Fraction_full_time_work',
                         'Fraction_part_time_work',
                         'Fraction_adult_school_or_full_time_student',
                         'Fraction_adult_degree',
                         'Fraction_employed_professional_occupations',
                         'Fraction_employed_elementary_occupations',
                         'Fraction_employed_skilled_trades']

cleaned_metric_names = {
    'Persons_per_Hectare':'Persons per hectare',
    'Persons_per_Household':  'Persons per household',
    'Fraction_Households_owned_by_tenants' : 'Households owned by tenants',
    'Fraction_Households_socially_rented' : 'Socially rented households',
    'Fraction_Households_privately_rented' : 'Privately rented households',
    'Fraction_of_persons_under_20':'Aged under 20',
    'Fraction_of_persons_20-29': 'Aged 20-29',
    'Fraction_of_persons_30-44': 'Aged 30-44',
    'Fraction_of_persons_45-64' : 'Aged 45-64',
    'Fraction_white' : 'White',
    'Fraction_UK_born' : 'UK born',
    'Fraction_adult_population_single' : 'Adult, single',
    'Fraction_married_partnership_widowed' : 'Married/partnersip/widowed',
    'Fraction_divorced_or_separated' : 'Divorced/separated',
    'Fraction_unemployed': 'Unemployed',
    'Fraction_full_time_work': 'Full time work',
    'Fraction_part_time_work': 'Part time work',
    'Fraction_adult_school_or_full_time_student': 'Adult at school/full time student',
    'Fraction_adult_degree': 'Adult with degree',
    'Fraction_employed_professional_occupations': 'Employed in professional occupation',
    'Fraction_employed_elementary_occupations': 'Employed in elementary occupation',
    'Fraction_employed_skilled_trades':'Employed in skilled trade' }

metrics_to_analyse = ['Fraction_white', 'Fraction_UK_born', 'Fraction_unemployed', 'Fraction_full_time_work']
def analyse_metrics(postcodes, metrics_to_consider = available_metrics(), verbose = False) : 
    
    """Determines how much a set of given metrics separate the given population from the average UK population."""
    
    oa_raw = opendata.load('/census/census_by_outputarea.csv')
    census_data = census.metrics.compute_metrics(oa_raw)
    oa_postcode_lookup = opendata.load('/census/postcode_outputarea_mapping.csv')
    cleaner = census.postcodes.PostcodeCleaner(oa_postcode_lookup['postcode'])
    postcodes =  cleaner(postcodes)
    postcodes_oas = pd.merge(pd.DataFrame({'postcode' : postcodes}), oa_postcode_lookup, how = 'inner', left_on = 'postcode', right_on = 'postcode')
    weight_column = 'Total_Population'
    weights = census_data[weight_column]
    joined = pd.merge(postcodes_oas, pd.DataFrame({'index': census_data.index, 'OA': census_data['OA']}), how = 'inner', on = 'OA')
    sample_indices = np.array(joined['index'])
    analyses = []

    for metric in metrics_to_consider:
        
        if verbose: 
            print("Analysing " + metric + ".")
            
        info = analyse_feature_split(census_data[metric], weights, sample_indices, bins = 3)
        info['metric'] = metric 
   
        if info['counts'][0] < info['counts'][1] < info['counts'][2]:
            info['direction'] = 'high'
        elif info['counts'][0] > info['counts'][1] > info['counts'][2]:
            info['direction'] = 'low'
        else:
            info['direction'] = 'middle'
        analyses.append(info)
    
    return analyses

def display_analysis(analyses) :
    
    """Displays the result of an analysis of metrics. """
    
    analyses.sort(key=lambda info: info['score'], reverse=True)
     
    filt = lambda s: s in ['high', 'low']

    metrics_descr = []
    scores = []
    directions = []
    colours = []
    
    for info in analyses:
        if filt(info['direction']):
            metrics_descr.append(cleaned_metric_names[info['metric']])
            scores.append(info['score'])
            directions.append(info['direction'])
            if info['direction'] == 'high':
                colours.append('rgb(19, 219, 82,0.8)')
            elif info['direction'] == 'low':
                colours.append('rgba(232,51,66,0.8)')
            else:
                colours.append('rgba(204,204,204,0.8)')
 
    
    
    data = [go.Bar(
            y=metrics_descr,
            x=scores,
            orientation = 'h',
            marker=dict(
            color=colours)
    )]
    
    annotations = []
    
    for yd in metrics_descr:
        annotations.append(dict(xref = 'x', yref = 'y',
                            x=5, y=yd,
                            text=yd,
                            xanchor = 'left',
                            font=dict(family='Arial', size=14,
                                      color='rgb(255, 255, 255)'), showarrow= False))
    layout = go.Layout(yaxis=dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ),
        xaxis = dict(title = 'Score') )
    layout['annotations'] = annotations
    fig = go.Figure(data=data, layout=layout)
    
    return fig
    
    


def postcode_map(postcodes):
    #The map
    geojson = opendata.load('/geojson/parliamentary_constituencies.json')
    folium_map = geoplot.centred_map(geojson, zoom_start = 11, location = (51.5074, 0.1278))
    plugins.ScrollZoomToggler().add_to(folium_map)

    #Postcode stuff
    oa_postcode_lookup = opendata.load('/census/postcode_outputarea_mapping.csv')
    postcode_to_coordinates = opendata.load('/census/ukpostcodes.csv')
    cleaner = census.postcodes.PostcodeCleaner(oa_postcode_lookup['postcode'])
    postcodes =  cleaner(postcodes)
    postcodes = [x for x in postcodes if not isinstance(x, float)]
    markers_to_add = postcode_to_coordinates.loc[postcode_to_coordinates['postcode'].isin(postcodes)][['latitude', 'longitude']]
    markers_to_add = markers_to_add.as_matrix()
    m = markers_to_add.shape[0]
    
    #Add the markers
    for i in range(m):
        folium.Marker(markers_to_add[i,:]).add_to(folium_map)
    
    
    return folium_map    