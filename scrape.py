

'''
C. McClintock 
Anarchist Library - web scraper 
'''

import requests # APIs
from bs4 import BeautifulSoup # web scraping
import re # regular expressions
import json # for writing json

'''
STEP ONE: get lists of urls 
thank you to the site for being so well indexed!
'''

urls = []
for page in range(1,24): # 23 pages
    # scrape page for index of documents
    URL = f"https://theanarchistlibrary.org/listing?sort=pages_asc&rows=500&page={page}"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find_all(class_="list-group-item clearfix")
    
    # match and return only urls 
    url_set = re.findall(r'href=[\"\'](.*?)[\"\']', str(results))
    urls.extend(url_set)


'''
STEP TWO: get titles, authors and works
thank you to Rohit for teaching me!
'''

lib = []

for i, url in enumerate(urls): 
    # run next set for now
    if i <= 5000: 
        continue # before, this was break

    try: 
        # get page content
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        # pull out fields of interest
        title = soup.find(id="text-title")
        author = soup.find(id="text-author")
        text = soup.find(id='thework')
        # add to dictionary
        entry = {
            'title': str(title.text), 
            'author': str(author.text),
            'text': str(text.text)    
            }
        # append to list
        lib.append(entry)

        print(f"Just added a work by {author.text}!") # for fun

    
    except: 
        print(f"Error loading {url}. Moving on to the next.")

    # write out to json when hits 1000
    if i > 0 and i % 1000 == 0:
        with open(f'lib-chunk-{i//1000}.json', 'w') as f:
            json.dump(lib, f, ensure_ascii=False)

        print(f"Bango! You've reached {i} documents. Pat yourself on the back. That's pretty cool.")

        lib = []

    



