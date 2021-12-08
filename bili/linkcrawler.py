#Simple assignment
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import re
import json
import time
import random

bvre = r"https:\/\/www\.bilibili\.com\/video\/(?P<bv>BV\w+)"


options = Options()
options.page_load_strategy = 'eager'

driver = Chrome('D:/zf-download/chromedriver',options=options)


#sub_areas = ['digital', 'application', 'computer_tech', 'industry', 'diy']
#sub_areas = ['science','social_science','humanity_history','business','campus','career','design','skill']
sub_areas = ['career','design','skill']

def getURL(area:str='tech', sub_area:str='digital', page=1):
    return f'https://www.bilibili.com/v/{area}/{sub_area}/#/all/click/0/{page}/2021-09-01,2021-11-30'

def getBV(url:str):
    return re.match(bvre,url).group('bv')

for sub_area in sub_areas:
    with open(f'data/{sub_area}.txt','a', encoding='utf-8') as dataFile:
        for page in range(300):
            driver.get(getURL(page=page+1, sub_area=sub_area, area='knowledge'))
            hrefs = []
            trial = 0
            while(len(hrefs)==0):
                time.sleep(1+random.random()*2)
                hrefs = driver.find_elements(by='css selector',value='.vd-list-cnt .title')
                trial += 1
                if trial>60:
                    driver.get(getURL(page=page+1, sub_area=sub_area))
                    trial = 0

            for href in hrefs:
                print(getBV(href.get_attribute('href')), href.text,file=dataFile)