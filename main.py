import os
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient
from langchain_openai import ChatOpenAI  # ✅ 최신 경로로 변경
from langchain.schema import HumanMessage

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OpenAI API 키를 찾을 수 없습니다.")

# OpenAI Embeddings 초기화
embedding_function = OpenAIEmbeddings()

# 기존 Chroma 벡터스토어 불러오기
chroma_store = Chroma(
    embedding_function=embedding_function,
    persist_directory="chroma_db",  # ✅ 기존 저장된 Chroma DB 경로
    collection_name="bakery_vector_store"  # ✅ 기존 컬렉션 이름
)

# Chroma check


client = PersistentClient(path="./chroma_db")  # Chroma 저장 경로 확인
#print(client.list_collections())  # 저장된 컬렉션 리스트 출력

collection = client.get_collection("bakery_vector_store")  # 올바른 컬렉션 이름으로 변경
print(collection.count())  # 저장된 벡터 개수 출력

docs = collection.get()  # 컬렉션의 모든 데이터 가져오기
print(f"저장된 벡터 개수: {len(docs['ids'])}")
#print(docs)  # 저장된 데이터 내용 확인


# =====================================
# 메인 코드: 사용자 성격 기반 빵집 추천
# =====================================

# 1. 테스트용 하드코딩된 사용자 성격 설명
personality_query = (
    "나는 요즘에 SNS에 핫한 베이커리를 방문하는 것을 좋아하고 유럽풍의 분위기를 좋아해. 반복적인 건 싫고 매일 새로운 곳에 가고싶어. "
)

# 2. Chroma 벡터스토어에서 사용자 성격과 유사한 빵집 문서를 검색 (예: 상위 3개)
similar_docs = chroma_store.similarity_search(personality_query, k=3)
print("판교의 빵집을 찾아다니고 있습니다....")

# 3. 추천 프롬프트 구성  
#    - 후보 빵집들의 정보를 나열하고, 재미있는 추천과 추천 이유를 요청하는 형태로 구성
recommendation_prompt = f"사용자의 성격: {personality_query}\n\n"
recommendation_prompt += "다음 빵집 후보들 중에서 사용자의 선호도나 성격을 분석해서 사용자가 가장 선호하고 좋아할만한 빵집을 하나 논리적인 근거를 들어서 추천해줘:\n"
for doc in similar_docs:
    recommendation_prompt += f"- {doc.page_content}\n"
# recommendation_prompt += "\n재미있는 추천과 함께, 왜 이 빵집을 추천했는지 상세하게 설명해줘."

# 4. ChatGPT API 호출 (추천 답변 생성)
llm = ChatOpenAI(temperature=0.7)
#recommendation = llm([HumanMessage(content=recommendation_prompt)])
recommendation = llm.invoke([HumanMessage(content=recommendation_prompt)])
print("판교의 빵집을 찾아다니고 있습니다....💨")

print("추천 결과:")
print(recommendation.content) #.invoke() 사용 시 결과 출력

# 5. 추천 이유만 별도로 물어보기  
#    (추천 결과의 빵집 정보를 기반으로 자세한 설명을 추가 요청)
explanation_prompt = (
    f"위의 추천 결과에 대해, 왜 해당 빵집을 추천했는지 구체적인 이유를 다시 한 번 설명해줘.\n\n"
    f"사용자 성격: {personality_query}\n\n"
    f"추천된 빵집 정보: {similar_docs[0].page_content}"
)

#explanation = llm([HumanMessage(content=explanation_prompt)])
explanation = llm.invoke([HumanMessage(content=explanation_prompt)])


print("\n추천 이유:")
print(explanation.content)