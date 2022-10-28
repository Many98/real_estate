import urllib
from urllib.request import urlopen
import json
from geojson import dump
import requests

for i in range(12, 23):
    url = "https://kriminalita.policie.cz/api/v2/downloads/20"+str(i)+"_554782.geojson"
    url = requests.get(url)
    text = url.text
    print(i)
    with open('criminality_'+str(i)+'.geojson', 'a', encoding='utf-8', errors='ignore') as f:
        dump(text, f)
