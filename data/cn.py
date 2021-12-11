import hanlp
from pathlib import Path
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)


from transformers import pipeline
unmasker = pipeline('fill-mask', model='data/bert-base-chinese')


import random
def cwb(sentence,p=0.5,maxrep=3, maxins=0):
  replaced = 0
  inserted = 0
  if len(sentence) == 0:
    return ''
  result = HanLP(sentence)
  words = result['tok/fine']
  #print(' '.join(words))
  wordtype= result['pos/ctb']
  for idx,word in enumerate(words[:]):
    for c in word:
      if c.isascii():
        wordtype[idx] = 'PU'
    if wordtype[idx] in ['AD','JJ','LC','NN','NR','NT','VA','VV'] and replaced < maxrep and random.random()<p:
      tgtchar = random.choice(word)
      words[idx] = word.replace(tgtchar,'[MASK]',1)
      subs = []
      for i in unmasker(''.join(words)):
        if not i['token_str'].isascii() and not i['token_str'] in ['，','。','；','“','”','：','！','？']:
          subs.append(i['token_str'])
      if len(subs) == 0:
        words[idx] = word
        continue
      words[idx] = word.replace(tgtchar, random.choice(subs))
      replaced += 1
      assert not '[MASK]' in ''.join(words)

    
    if wordtype[idx] in ['AD','JJ','LC','NN','NR','NT','VA','VV'] and inserted < maxins and random.random()<p:
      words[idx] = words[idx]+'[MASK]' if random.random()<0.5 else '[MASK]'+words[idx]
      subs = []
      for i in unmasker(''.join(words)):
        if not i['token_str'].isascii() and not i['token_str'] in ['，','。','；','“','”','：','！','？']:
          subs.append(i['token_str'])
      if len(subs) == 0:
        words[idx] = c
        continue
      words[idx] = words[idx].replace('[MASK]', random.choice(subs))
      inserted += 1

  return ''.join(words)


#sub_areas = ['science','social_science','humanity_history','business','campus','career','design','skill']
sub_areas = ['douga','music','dance','game','knowledge','tech','sports','car','life','food','animal','fashion','information','ent']

PROB = 0.8

savePath = f'data/augdata/p{PROB}/'
Path(savePath).mkdir(exist_ok=True,parents=True)
dataPath = 'data/'


for sub_area in sub_areas:
  print(f'Subarea: {sub_area}')
  with open(dataPath+sub_area+'.txt', encoding='utf-8') as dataFile:
    dataText = dataFile.read().split('\n')
    dataText = [i[13:] for i in dataText]
    augText1 = []
    augText2 = []
    for idx,t in enumerate(dataText):
      print(f'\rAugmentation {idx}/{len(dataText)}',end='')
      augText1.append(cwb(t,p=PROB))
      augText2.append(cwb(t,p=PROB))

    with open(savePath+sub_area+'1.txt','w',encoding='utf-8') as saveFile:
      saveFile.write('\n'.join(augText1))
    with open(savePath+sub_area+'2.txt','w',encoding='utf-8') as saveFile:
      saveFile.write('\n'.join(augText2))

    print(f'Subarea: {sub_area} saved')






