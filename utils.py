
import pandas as pd
import numpy as np
import folium 
from lxml import etree
import json
import requests
from geopy.geocoders import Nominatim


def query_term(term):
    '''
    Definition of a function that will query the geonames API and return the name of a Canton if it appears in the search.

    '''
    
    if term != '-' and term != 'de' and term != 'di' and term != 'und' and term != 'of':
        

        term_value = term

        tree = etree.parse("http://api.geonames.org/search?name=%s&maxRows=200&country=CH&username=thomvett" %  (term_value))
        for user in tree.xpath("/geonames/geoname/fcode"):
            #print(user.text)
            if user.text == "ADM1":

                for i in user.itersiblings(preceding=True):
                    if i.tag == 'name':
                        return i.text
                    
        return 'no canton found'
    else:
        return 'no canton found'
    
    
def query_city(term):
    '''
    Returns city from Uni list
    
    '''
    lat_lng = []
    if term != '-' and term != 'de' and term != 'di' and term != 'und' and term != 'of' and term != '':
        

        term_value = term

        tree = etree.parse("http://api.geonames.org/search?name_equals=%s&maxRows=100&country=CH&username=thomvett" %(term_value))
        for user in tree.xpath("/geonames/geoname/fcode"):
            #print(user.text)
            if user.text == "PPL" or user.text =="PPLA":

                for i in user.itersiblings(preceding=True):
                    if i.tag == 'lng':
                        lat_lng.append(i.text)
                    if i.tag == 'lat':
                        lat_lng.append(i.text)
                        
                        
                return lat_lng
                        
                    
        return 'no city found'
    else:
        return 'no city found'    
    
def coordinate(test):
    '''
    This returns the geographical coordinate
    '''
    coord=[]
    for i in test:
        geolocator = Nominatim()
        loc=geolocator.geocode(i)
        coor1 = [loc.latitude,loc.longitude]
    
    
        coord.append(coor1)   
    return coord
    