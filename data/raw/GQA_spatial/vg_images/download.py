
import json
import wget
from tqdm import tqdm
urls = json.load(open('add_urls.json'))

for url in tqdm(urls):
    try:
        wget.download(url)
    except:
        pass

