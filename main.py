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

#collection = client.get_collection("bakery_vector_store")  # 올바른 컬렉션 이름으로 변경
#print(collection.count())  # 저장된 벡터 개수 출력

#docs = collection.get()  # 컬렉션의 모든 데이터 가져오기
#print(f"저장된 벡터 개수: {len(docs['ids'])}")
#print(docs)  # 저장된 데이터 내용 확인


class AIModel():
    def request(self, personality_query):

        # 2. Chroma 벡터스토어에서 사용자 성격과 유사한 빵집 문서를 검색 
        similar_docs = chroma_store.similarity_search(personality_query, k=3)

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


        # for i, doc in enumerate(similar_docs):
        #     explanation_prompt += f"{i+1}. {doc.page_content}\n\n"

        results = []
        
        # ✅ 빵집 정보 추출 (최대 3개)
        for i, doc in enumerate(similar_docs):
            text = doc.page_content
            print(text, '\n')

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
            
            explanation_prompt = (
            f"이 빵집이 사용자의 성격, 기분, 상황과 어떻게 연결되는지 논리적으로 개인화된 설명해 주세요.\n"
            f"추천 과정이 억지스럽지 않고, 성격과 빵집의 특징이 자연스럽게 연결되도록 해주세요.\n"
            f"1. 사용자의 대화에서 중요한 특징을 뽑아주세요.\n"
            f"2. 해당 특징이 어떤 이유로 이 빵집과 잘 맞는지 구체적인 근거를 제시해주세요.\n"
            f"3. 마지막으로, 듣는 사람이 기분 좋게 느낄 수 있도록 친절하고 예쁜 말투로 설명해주세요.\n"
            f"4. 설명은 3~4줄 정도로 간결하게 해주세요.\n\n"
            f"논리가 있다는 얘기는 할 필요 없고 그냥 논리적인 접근만 한 줄로 주면 돼요.\n"
            f"explanation을 빵집이름으로 시작하게 메세지를 작성해주세요 .\n"
            f"중복 없이 추천해주세요.\n"
            f"사용자 성격: {personality_query}\n"
            f"{i+1}.{text}\n\n"
        )
            
            explanation = llm.invoke([HumanMessage(content=explanation_prompt)])

            results.append({
                "name": bakery_name,
                "score": overall_score,
                "taste_score": taste_score,
                "price_score": price_score,
                "cs_score": cs_score,
                "address": address,
                "keywords": unique_keywords,
                "explanation": explanation.content
            })

        return results
