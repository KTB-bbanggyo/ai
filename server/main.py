import os
import re

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from chromadb import PersistentClient

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API 키를 찾을 수 없습니다.")

chroma_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",  
    collection_name="bakery_vector_store"
)

client = PersistentClient(path="./chroma_db") 
collection = client.get_collection("bakery_vector_store") 
docs = collection.get()  

class AIModel():
    def request(self, personality_query):

        similar_docs = chroma_store.similarity_search(personality_query, k=1)

        llm = ChatOpenAI(temperature=0.7)

        explanation_prompt = (
            f"해당 빵집이 내가 입력한 성격과 무슨 관계가 있는지 세 줄 이내로 설명해.\n"
            f"말장난이나 빵집의 성격을 사용해서 억지같지만 나름의 논리가 있는 과정을 거쳤다고 설명해.\n"
            f"논리가 있다는 얘기는 할 필요 없고 그냥 듣는 사람이 재밌고 납득가게 한 줄로 주면 돼. \n"
            f"그리고 손님에게 접대하듯이 듣는 사람이 기분 좋게 예쁘게 말하고 ~~해요 체로 얘기해.\n\n"
            f"사용자 성격: {personality_query}\n"
            f"추천된 빵집 정보: {similar_docs[0].page_content}"
        )
        explanation = llm.invoke([HumanMessage(content=explanation_prompt)])

        text = similar_docs[0].page_content
        print(text)

        bakery_name_match = re.search(r'빵집 이름:\s*(.+)', text)
        bakery_name = bakery_name_match.group(1).strip() if bakery_name_match else None

        overall_score_match = re.search(r'총점\s*([\d.]+)', text)
        overall_score = overall_score_match.group(1).strip() if overall_score_match else None

        taste_score_match = re.search(r'맛\s*([\d.]+)', text)
        taste_score = taste_score_match.group(1).strip() if taste_score_match else None

        price_score_match = re.search(r'가격\s*([\d.]+)', text)
        price_score = price_score_match.group(1).strip() if price_score_match else None

        address_match = re.search(r'주소:\s*(.+)', text)
        address = address_match.group(1).strip() if address_match else None

        cs_score_match = re.search(r'고객서비스\s*([\d.]+)', text)
        cs_score = cs_score_match.group(1).strip() if cs_score_match else None

        review_keywords = re.findall(r'키워드:\s*([^)]+)', text)
        unique_keywords = sorted({kw.strip() for group in review_keywords for kw in group.split(",")})


        result = {
            "name": bakery_name,
            "score": overall_score,
            "taste_score": taste_score,
            "price_score": price_score,
            "cs_score": cs_score,
            "address": address,
            "keywords": unique_keywords,
            "explanation": explanation.content
        }

        return result