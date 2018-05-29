import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import pdfkit
http = httplib2.Http()
status, response = http.request('http://www.deeplearningbook.org/')

pdfkit.from_url('google.com', 'out.pdf')

for link in BeautifulSoup(response, parseOnlyThese=SoupStrainer('a')):
         if link.has_attr('href'):
             if 'contents' in link['href']:
                 _l = link['href']
                 if 'contents/' in _l:
                     url = 'http://www.deeplearningbook.org/' + _l
                     print(url)
                     print('/Users/joel.stevens/Desktop/' + _l.replace('contents/','').replace('.html','.pdf'))
                     pdfkit.from_url(url, '/Users/joel.stevens/Desktop/' + _l.replace('contents/','').replace('.html','.pdf'))
