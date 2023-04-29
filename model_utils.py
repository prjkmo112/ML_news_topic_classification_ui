import re
from konlpy.tag import Kkma
from konlpy.utils import pprint
from nltk.corpus import stopwords
import utils as _utils

kkma = Kkma()
utils = _utils.Utils()


class modelUtils:
    def __init__(self) -> None:
        pass

    def dataPreprocessing(self, title):
        ## 형태소 분리 (일단, 특수문자만 처리해보고 결과가 너무 안좋으면 형태소 분석기)
        # title = re.sub(r"…|!|~|∼|·", " ", title).strip()
        
        # 형태소 분리
        # 특수기호 제거
        kkma_title = kkma.pos(title)
        kkma_title_list = map(lambda v: v[0], filter(lambda v: re.compile(r"^(?:N|V)").match(v[1]), kkma_title))

        new_title = ' '.join(kkma_title_list)

        # 불용어 제거
        stops_en = list(set(stopwords.words("english")))
        stops_ko = utils.stopwords("ko2")
        stops = stops_en + stops_ko

        title_words = new_title.split()
        title_words = [tt for tt in title_words if tt not in stops]
        clean_title = ' '.join(title_words)

        return clean_title
    

    