import os

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


# embedding_function = OpenAIEmbeddings()
chroma_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",  # 기존 저장된 Chroma DB 경로
    collection_name="bakery_vector_store"  # 기존 컬렉션 이름
)

client = PersistentClient(path="./chroma_db") 

collection = client.get_collection("bakery_vector_store")  # 올바른 컬렉션 이름으로 변경
print(collection.count())  # 저장된 벡터 개수 출력

docs = collection.get()  # 컬렉션의 모든 데이터 가져오기
print(f"저장된 벡터 개수: {len(docs['ids'])}")
#print(docs)  # 저장된 데이터 내용 확인


class AIModel():
    def request(personality_query):

        # 2. Chroma 벡터스토어에서 사용자 성격과 유사한 빵집 문서를 검색 (예: 상위 3개)
        similar_docs = chroma_store.similarity_search(personality_query, k=3)
        print("판교의 빵집을 찾아다니고 있습니다....")

        # 3. 추천 프롬프트 구성  
        #    - 후보 빵집들의 정보를 나열하고, 재미있는 추천과 추천 이유를 요청하는 형태로 구성
        recommendation_prompt = f"사용자의 성격: {personality_query}\n\n"
        recommendation_prompt += "다음 빵집 후보들 중에서 사용자에게 가장 어울리는 서로 다른 빵집을 세 개 추천해줘.\n"
        recommendation_prompt += "반드시 서로 다른 빵집이어야 하고, 반드시 문서에 있는 빵집이어야 해."

        for doc in similar_docs:
            recommendation_prompt += f"- {doc.page_content}\n"
        recommendation_prompt += "\n설명은 필요없고 빵집 이름이랑 별점 알려줘. 양식은 제목 \n 총점 : nn 맛 : nn 가격 : nn 고객서비스 : nn"

        # 4. ChatGPT API호출
        llm = ChatOpenAI(temperature=0.7)
        recommendation = llm.invoke([HumanMessage(content=recommendation_prompt)])
        print("판교의 빵집을 찾아다니고 있습니다....💨")

        print("추천 결과:")
        print(recommendation.content)

        explanation_prompt = (
            f"위의 추천 결과에 대해, 해당 빵집이 내가 입력한 성격과 무슨 관계가 있는지 한 줄로 설명해줘.\n\n"
            f"사용자 성격: {personality_query}\n\n"
            f"추천된 빵집 정보: {similar_docs[0].page_content}"
        )
        explanation = llm.invoke([HumanMessage(content=explanation_prompt)])

        print("\n추천 이유:")
        print(explanation.content)