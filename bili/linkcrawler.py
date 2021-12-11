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
#sub_areas = ['career','design','skill']
#areas = ['douga','music','dance','game','knowledge','tech','sports','car','life','food','animal','fashion','information','ent']
areas = ['game','knowledge','tech','sports','car','food','animal','fashion','information','ent']

def getURL(area:str='tech', sub_area:str='', page=1, period=['09','11']):
    if len(sub_area):
        return f'https://www.bilibili.com/v/{area}/{sub_area}/#/all/click/0/{page}/2021-{period[0]}-01,2021-{period[1]}-30'
    else:
        return f'https://www.bilibili.com/v/{area}/'

def getBV(url:str):
    return re.match(bvre,url).group('bv')

def get_sub_areas(hrefs):
    return [i.get_attribute('href').split('/')[-2] for i in hrefs]



for area in areas:
    with open(f'data/{area}.txt','a', encoding='utf-8') as dataFile:
        driver.get(getURL(area=area))
        hrefs = []
        while not len(hrefs):
            hrefs = driver.find_elements(by='css selector',value='#subnav a')[1:]
        sub_areas = get_sub_areas(hrefs)
        for sub_area in sub_areas:
            if sub_area in ['match']:
                break
            empty = False
            for page in range(50):
                driver.get(getURL(page=page+1, sub_area=sub_area, area=area))
                hrefs = []
                trial = 0
                while(len(hrefs)==0):
                    time.sleep(1+random.random()*2)
                    if len(driver.find_elements(by='css selector',value='.empty')) or len(driver.find_elements(by='css selector',value='.error-404')):
                        empty = True
                        break
                    hrefs = driver.find_elements(by='css selector',value='.vd-list-cnt .title')
                    trial += 1
                    if trial>60:
                        driver.get(getURL(page=page+1, sub_area=sub_area))
                        trial = 0
                if empty:
                    break
                for href in hrefs:
                    print(getBV(href.get_attribute('href')), href.text,file=dataFile)