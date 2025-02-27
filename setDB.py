import json, os
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API 키를 찾을 수 없습니다.")

# ✅ MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")

# ✅ 데이터베이스 및 컬렉션 선택
db = client["bakery_db"]
collection = db["bakery_data"]

# =====================================
# ✅ MongoDB에서 빵집 데이터 불러오기 및 LangChain Document 생성
# =====================================
bakery_data = list(collection.find())
documents = []

for data in bakery_data:
    title = data.get("title", "제목 없음")
    scores = data.get("scores", {})
    total_score = scores.get("total_score", "N/A")
    taste_score = scores.get("taste_score", "N/A")
    price_score = scores.get("price_score", "N/A")
    cs_score = scores.get("cs_score", "N/A")

    # 각 리뷰 정보를 결합 (리스트에 있는 요소가 dict인지 확인)
    review_texts = []
    for review in data.get("reviews", []):
        if isinstance(review, dict):  # ✅ 리뷰가 dict일 때만 처리
            content = review.get("content", "")
            score = review.get("score", "")
            keywords = ", ".join(review.get("keywords", []))
            review_texts.append(f"리뷰: {content} (평점: {score}, 키워드: {keywords})")
        else:
            print(f"⚠️ 잘못된 리뷰 데이터 감지: {review}")  # 문제 있는 리뷰 확인용

    reviews_combined = "\n".join(review_texts)

    address = data.get("address")


    # 하나의 문서로 결합
    content_text = (
        f"빵집 이름: {title}\n"
        f"평점: 총점 {total_score}, 맛 {taste_score}, 가격 {price_score}, 고객서비스 {cs_score}, 주소: {address}\n"
        f"{reviews_combined}"
    )

    # ✅ LangChain Document 생성
    documents.append(Document(
        page_content=content_text,
        metadata={"_id": str(data.get("_id")), "title": title}
    ))

print(f"✅ LangChain 문서 {len(documents)}개 생성 완료!")

# =====================================
# ✅ Chroma 벡터스토어 생성 (임베딩 생성)
# =====================================
embedding_function = OpenAIEmbeddings()
persist_dir = "./chroma_db"

chroma_store = Chroma.from_documents(
    documents, embedding_function,
    collection_name="bakery_vector_store",
    persist_directory=persist_dir  # 영구 저장
)

print("✅ ChromaDB에 벡터 저장 완료!")

# ===========