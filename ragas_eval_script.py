import os
import time
import requests
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall
)

from langchain_google_genai import ChatGoogleGenerativeAI

# ========== Load Environment ==========
load_dotenv()
GEMINI_API_KEYS = os.getenv("GOOGLE_API_KEYS", "").split(",")
API_KEY = os.getenv("API_KEY")
DOCUMENT_URL = os.getenv("DOCUMENT_URL")
API_URL = "http://localhost:8000/api/v1/hackrx/run"

# ========== Key Rotation ==========
class GeminiKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = -1  # Start at -1 so first rotate gives 0

    def rotate_key(self):
        self.index = (self.index + 1) % len(self.keys)
        return self.keys[self.index]

    def get_key(self):
        return self.keys[self.index]

key_manager = GeminiKeyManager(GEMINI_API_KEYS)

def get_llm():
    api_key = key_manager.rotate_key()
    print(f"[Gemini] Using API Key: {api_key[:8]}... (rotated)")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
        timeout=None,
        max_retries=0
    )

# ========== Sample Queries ==========
sample_queries = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What expenses are covered under in-patient treatment in the National Parivar Mediclaim Plus Policy?",
    "Are pre-hospitalisation expenses covered?",
    "Is day care treatment covered under this policy?",
    "Are AYUSH treatments covered in the National Parivar Mediclaim Plus Policy?"
]

expected_responses = [
    "The Grace Period for payment of the premium shall be thirty days.",
    "The policy covers room and ICU charges, medical practitioner fees, anesthesia, blood, oxygen, operation theatre charges, surgical appliances, medicines and drugs, diagnostic procedures, prosthetics, dental treatment due to injury, plastic surgery due to disease or injury, hormone replacement therapy if medically necessary, vitamins and tonics forming part of treatment, and circumcision if medically necessary.",
    "Yes, medical expenses up to 30 days before hospitalisation are covered if they relate to the same condition and the in-patient claim is admissible.",
    "Yes, day care treatments requiring hospitalisation for less than 24 hours are covered, including pre and post hospitalisation expenses, but not if done in the outpatient department.",
    "Yes, inpatient treatments under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy are covered up to the Sum Insured limit as per the Policy Schedule in any AYUSH hospital."
]

# ========== Get Model Answers ==========
dataset_list = []
for question, ground_truth in zip(sample_queries, expected_responses):
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "documents": DOCUMENT_URL,
                "questions": [question]
            }
        )
        response.raise_for_status()
        response_json = response.json()

        answer = response_json["answers"][0]
        chunks = response_json["chunks"][0]

        dataset_list.append({
            "user_input": question,
            "retrieved_contexts": chunks,
            "response": answer,
            "reference": ground_truth
        })

    except Exception as e:
        print(f"❌ Error fetching response for '{question}': {e}")

# ========== Print Response Summary ==========
print("\n--- Comparison of Responses ---")
for i, data in enumerate(dataset_list):
    print(f"\nExample {i+1}:")
    print(f"Q: {data['user_input']}")
    print(f"🔹 Model Answer:\n{data['response']}")
    print(f"🔸 Expected Answer:\n{data['reference']}")
    print(f"📚 Retrieved Context:\n{data['retrieved_contexts'][:300]}...")

# ========== Evaluation Dataset ==========
evaluation_dataset = EvaluationDataset.from_list(dataset_list)

# ========== Retryable Evaluation ==========
def safe_evaluate(dataset, metrics):
    max_attempts = len(GEMINI_API_KEYS)
    for attempt in range(max_attempts):
        try:
            llm = LangchainLLMWrapper(get_llm())
            result = evaluate(dataset=dataset, metrics=metrics, llm=llm)
            return result
        except Exception as e:
            print(f"⚠️ Evaluation failed on attempt {attempt + 1}: {e}")
            time.sleep(1)
    raise RuntimeError("❌ All Gemini API keys failed.")

# ========== Run Evaluation ==========
result = safe_evaluate(
    dataset=evaluation_dataset,
    metrics=[FactualCorrectness(), Faithfulness(), LLMContextRecall()]
)

# ========== Output ==========
print("\n--- Evaluation Results ---")
print(result)

print("\n\n--- Comparison of Responses ---")
for i, data in enumerate(dataset_list):
    print(f"\nExample {i+1}:")
    print(f"Q: {data['user_input']}")
    print(f"🔹 Model Answer:\n{data['response']}")
    print(f"🔸 Expected Answer:\n{data['reference']}")
    print(f"📚 Retrieved Context:\n{data['retrieved_contexts'][:300]}...")
