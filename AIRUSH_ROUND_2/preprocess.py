import pandas as pd
import re

from soynlp.normalizer import * # 복수 글자 반복을 줄여주는 패키지
from  hanspell import spell_checker
from typing import List

from tqdm import tqdm
import time

def remove_dup(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    중복 문장 제거 후 Pandas 형태로 Return
    
    => Context와 Label의 순서를 맞춰주면서 중복 문장 제거하기 위해 활용
    """
    new_text = []
    new_label = []
    for i in range(len(dataset)):
        ctxt = dataset['contents'][i]
        label = dataset['label'][i]
        if ctxt not in new_text:
            new_text.append(ctxt)
            new_label.append(label)
    
    new_dataset = pd.DataFrame({'contents': new_text,
                                'label': new_label})
    return new_dataset

def spell_check_using_hanspell(text: List, max_length=512, is_training=True) -> List:
    """
    hanspell library를 활용하여 spell check

    Tokenizer의 Max_Length를 기준으로 문장을 일정 길이로 자른 후 spell check 숳갱
    """
    new_text = []
    timer = 0
    for t in tqdm(text):
        timer += 1
        if is_training and timer % 2000 == 0:
            time.sleep(10)

        ctxt = ' '.join(t.split()[:max_length])

        k=0
        replaced_text = ''
        while True:
            origin_text = ctxt[k:k+500]
            check_text = spell_checker.check(origin_text)
            check_text = check_text.checked
            replaced_text += check_text
            
            k += 500
            if len(origin_text) < 500:
                break
        new_text.append(replaced_text)
    return new_text

def remove_email(texts) :
    '''
    이메일을 제거한다. (이메일은 개인정보가 될 수 있으므로, 자연어처리를 진행할 때 반드시 없애주도록 한다.)
    '홍길동 abc@gmail.com 연락주세요!' -> '홍길동  연락주세요!'
    '''
    preprocessed_text = []
    for text in texts :
        text = re.sub('[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', text).strip()
        if text :
            preprocessed_text.append(text)
    return preprocessed_text

def remove_user_mention(texts) :
    '''
    유저에 대한 멘션(@) 태그를 제거한다. (@역시 개인정보를 포함하고 있을 수 있으므로 제거하도록 한다.)
    '@홍길동 감사합니다!' -> ' 감사합니다!'
    '''
    preprocessed_text = []
    for text in texts :
        text = re.sub(r"@\w+", "", text).strip()
        if text :
            preprocessed_text.append(text)
    return preprocessed_text

def remove_url(texts) :
    '''
    URL을 제거한다. (URL 역시 개인정보 혹은 우리가 가져와서는 안되는 중요한 정보(주소 등)을 포함하고 있을 위험이 있으므로 반드시 제거해주도록 한다.)
    '주소: www.naver.com' -> '주소: '
    '''
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
        text = re.sub(r"pic\.(\w+\.)+\S*", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def remove_bad_char(texts):
    """
    문제를 일으킬 수 있는 문자들을 제거합니다.
    (한국어의 특성상 생기는 문제들이며, 한국어 데이터를 크롤링할 때, 아래의 bad_chars에 있는 것들 처럼 알 수 없는 기호들이 크롤링 되기도 한다.)
    (이러한 문자들로 인하여 vocab을 생성할 때 에러가 발생할 수 있다.)
    """
    bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
    preprcessed_text = []
    for text in texts:
        for bad_char in bad_chars:
            text = text.replace(bad_char, bad_chars[bad_char])
        text = re.sub(r"[\+á?\xc3\xa1]", "", text)
        if text:
            preprcessed_text.append(text)
    return preprcessed_text

def remove_press(texts):
    """
    뉴스 데이터를 가져왔을 경우, 언론 정보를 제거한다.
    ``홍길동 기자 (연합뉴스)`` -> ````
    ``(이스탄불=연합뉴스) 하채림 특파원 -> ````
    """
    re_patterns = [
        r"\([^(]*?(뉴스|경제|일보|미디어|데일리|한겨례|타임즈|위키트리)\)",
        r"[가-힣]{0,4} (기자|선임기자|수습기자|특파원|객원기자|논설고문|통신원|연구소장) ",  # 이름 + 기자
        r"[가-힣]{1,}(뉴스|경제|일보|미디어|데일리|한겨례|타임|위키트리)",  # (... 연합뉴스) ..
        r"\(\s+\)",  # (  )
        r"\(=\s+\)",  # (=  )
        r"\(\s+=\)",  # (  =)
    ]

    preprocessed_text = []
    for text in texts:
        for re_pattern in re_patterns:
            text = re.sub(re_pattern, "", text).strip()
        if text:
            preprocessed_text.append(text)    
    return preprocessed_text

def remove_copyright(texts):
    """
    뉴스 데이터를 가져왔을 경우, 뉴스 내 포함된 저작권 관련 텍스트를 제거한다.
    ``(사진=저작권자(c) 연합뉴스, 무단 전재-재배포 금지)`` -> ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` 
    """
    re_patterns = [
        r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
        r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
    ]
    preprocessed_text = []
    for text in texts:
        for re_pattern in re_patterns:
            text = re.sub(re_pattern, "", text).strip()
        if text:
            preprocessed_text.append(text)    
    return preprocessed_text

def remove_photo_info(texts):
    """
    뉴스 데이터를 가져왔을 경우, 뉴스 내 포함된 이미지에 대한 label을 제거합니다.
    ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` -> ````
    ``(출처=청주시)`` -> ````
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def remove_useless_bracket(texts) :
    '''
    위키피디아 전처리를 위한 함수이다.
    괄호 내부에 의미가 없는 정보를 제거한다.
    아무런 정보를 포함하고 있지 않다면, 괄호를 통채로 제거한다.
    ``수학(,)`` -> ``수학``
    ``수학(數學,) -> ``수학(數學)``
    '''
    bracket_pattern = re.compile(r"\((.*?)\)")
    preprocessed_text = []
    for text in texts :
        modi_text = ''
        text = text.replace('()', '') # 수학 () -> 수학
        brackets = bracket_pattern.search(text)
        if not brackets : # 쓸모없는 bracket이 없을 경우
            if text :
                preprocessed_text.append(text)
                continue
        
        replace_brackets = {} 
        # key : 원본 문장에서 고쳐야하는 index
        # value : 고쳐져야 하는 값
        # e.g. {'2,8': '(數學)','34,37': ''}
        while brackets :
            index_key = str(brackets.start()) + ',' + str(brackets.end()) # '2,8', '34,37'과 같이 만들어서 key로 넘겨준다.
            bracket = text[brackets.start() + 1 : brackets.end() - 1] # 괄호 안에 있는 문자
            infos = bracket.split(',')
            modi_infos = []
            for info in infos :
                info = info.strip()
                if len(info) > 0 :
                    modi_infos.append(info) # info가 빈칸이 아니면 modi_infos에 넣어준다.
            if len(modi_infos) > 0 : # modi_infos가 원소를 가지고 있으면 index_key에 대한 value를 그 원소로, 없으면 빈칸으로 준다.
                replace_brackets[index_key] = '(' + ', '.join(modi_infos) + ')'
            else :
                replace_brackets[index_key] = ''
            brackets = bracket_pattern.search(text, brackets.start() + 1)
        end_index = 0
        for index_key in replace_brackets.keys() :
            start_index = int(index_key.split(',')[0])
            modi_text += text[end_index:start_index] # 괄호 전까지의 텍스트를 넣어준다.
            modi_text += replace_brackets[index_key] # 수정된 괄호 내용을 텍스트에 넣어준다.
            end_index = int(index_key.split(',')[1])
        modi_text += text[end_index:] # 괄호 이후 텍스트를 넣어준다.
        modi_text = modi_text.strip()

        if modi_text :
            preprocessed_text.append(modi_text)
    return preprocessed_text

def remove_repeat_char(texts):
    '''
    여러 글자가 반복될 경우 그 반복 수를 제한해준다.
    '와하하하하하하핫' (num_repeats = 2) -> '와하하핫'
    '''
    preprocessed_text = []
    for text in texts:
        text = repeat_normalize(text, num_repeats=2).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def clean_punc(texts):
    '''
    기호들을 일반화한다. (따옴표 통일, 루트기호를 sqrt로 변경 등)
    punct_mapping dictionary의 내용들을 바탕으로 수정해준다.
    '''
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    preprocessed_text = []
    for text in texts:
        for p in punct_mapping:
            text = text.replace(p, punct_mapping[p])
        text = text.strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def remove_repeated_spacing(texts):
    """
    두 개 이상의 연속된 공백을 하나로 치환합니다.
    ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text

def preprocessing(context):
    context = [context]
    context = remove_email(context)
    context = remove_user_mention(context)
    context = remove_url(context)
    context = remove_bad_char(context)
    context = remove_press(context)
    context = remove_copyright(context)
    context = remove_photo_info(context)
    context = remove_useless_bracket(context)
    context = remove_repeat_char(context)
    context = clean_punc(context)
    context = remove_repeated_spacing(context)
    return context[0]