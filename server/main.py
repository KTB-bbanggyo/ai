import os
import json
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


# # embedding_function = OpenAIEmbeddings()
# chroma_store = Chroma(
#     embedding_function=OpenAIEmbeddings(),
#     persist_directory="chroma_db",  # 기존 저장된 Chroma DB 경로
#     collection_name="bakery_vector_store",  # 기존 컬렉션 이름
#     openai_proxy="http://krmp-proxy.9rum.cc:3128"
# )

chroma_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",  # 기존 저장된 Chroma DB 경로
    collection_name="bakery_vector_store"
)

client = PersistentClient(path="./chroma_db") 

collection = client.get_collection("bakery_vector_store")  # 올바른 컬렉션 이름으로 변경
print(collection.count())  # 저장된 벡터 개수 출력

docs = collection.get()  # 컬렉션의 모든 데이터 가져오기
print(f"저장된 벡터 개수: {len(docs['ids'])}")
#print(docs)  # 저장된 데이터 내용 확인


class AIModel():
    def request(self, personality_query):

        # 2. Chroma 벡터스토어에서 사용자 성격과 유사한 빵집 문서를 검색 
        similar_docs = chroma_store.similarity_search(personality_query, k=1)

        # # 3. 추천 프롬프트 구성  
        # recommendation_prompt = f"사용자의 성격: {personality_query}\n\n"
        # recommendation_prompt += "다음 빵집 후보들 중에서 사용자에게 가장 어울리는 빵집을 단 하나만 추천해줘.\n"
        # recommendation_prompt += "반드시 서로 다른 빵집이어야 하고, 반드시 문서에 있는 빵집이어야 해.\n"

        # for doc in similar_docs:
        #     recommendation_prompt += f"- {doc.page_content}\n"
        # recommendation_prompt += "설명은 필요없고 빵집 이름이랑 별점 알려줘. 양식은 제목 \n 총점 : nn 맛 : nn 가격 : nn 고객서비스 : nn"

        # # 4. ChatGPT API호출
        llm = ChatOpenAI(temperature=0.7)
        # recommendation = llm.invoke([HumanMessage(content=recommendation_prompt)])


        explanation_prompt = (
            f"해당 빵집이 내가 입력한 성격과 무슨 관계가 있는지 세 줄 이내로 설명해.\n"
            f"말장난이나 빵집의 성격을 사용해서 억지같지만 나름의 논리가 있는 과정을 거쳤다고 설명해.\n"
            f"논리가 있다는 얘기는 할 필요 없고 그냥 논리적인 접근만 한 줄로 주면 돼.\n"
            f"그리고 손님에게 접대하듯이 듣는 사람이 기분 좋게 예쁘게 말하고 ~~해요 체로 얘기해.\n\n"
            f"사용자 성격: {personality_query}\n"
            f"추천된 빵집 정보: {similar_docs[0].page_content}"
        )
        explanation = llm.invoke([HumanMessage(content=explanation_prompt)])


        text = similar_docs[0].page_content

        # 빵집 이름 추출
        bakery_name_match = re.search(r'빵집 이름:\s*(.+)', text)
        bakery_name = bakery_name_match.group(1).strip() if bakery_name_match else None

        # 총점 추출 (평점: 총점 부분)
        overall_score_match = re.search(r'총점\s*([\d.]+)', text)
        overall_score = overall_score_match.group(1).strip() if overall_score_match else None

        taste_score_match = re.search(r'맛\s*([\d.]+)', text)
        taste_score = overall_score_match.group(1).strip() if taste_score_match else None

        price_score_match = re.search(r'가격\s*([\d.]+)', text)
        price_score = overall_score_match.group(1).strip() if price_score_match else None

        cs_score_match = re.search(r'고객서비스\s*([\d.]+)', text)
        cs_score = overall_score_match.group(1).strip() if cs_score_match else None

        review_keywords = re.findall(r'키워드:\s*([^)]+)', text)

        unique_keywords = sorted({kw.strip() for group in review_keywords for kw in group.split(",")})


        result = {
            "name": bakery_name,
            "score": overall_score,
            "taste_score": taste_score,
            "price_score": price_score,
            "cs_score": cs_score,
            "keywords": review_keywords,
            "explanation": explanation.content
        }

        unique_keywords = set()
        for keyword_group in result["keywords"]:
            keywords = [kw.strip() for kw in keyword_group.split(",")]
            unique_keywords.update(keywords)

        # 집합을 리스트로 변환 및 정렬(선택사항)
        final_keywords = sorted(list(unique_keywords))

        # data 딕셔너리의 keywords 필드 업데이트
        result["keywords"] = final_keywords

        print(result)
        return result