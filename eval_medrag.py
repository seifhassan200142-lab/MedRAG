"""
MedRAG — eval_medrag.py (Chainlit Fixed Version)
=========================================
Automated Evaluation Pipeline for MedRAG Chainlit Backend.
Now integrates Hybrid Retrieval to satisfy MedRAGChain dependencies.
"""

import os
import sys
import time
import asyncio
import argparse
import warnings
import pandas as pd
import numpy as np

# MLflow Components
import mlflow

warnings.filterwarnings('ignore')
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# إضافة المسار الحالي للتأكد من قراءة فولدر src بشكل سليم
sys.path.append(os.getcwd())

try:
    # استدعاء الموديولز والـ Pipeline كامل من مشروعك كـ Black-Box
    from src.generation.llm_chain import MedRAGChain
    from src.retrieval.searcher import HybridSearcher
    from src.retrieval.expansion import QueryAugmentor
except ImportError as e:
    print("❌ Error: Could not find MedRAG modules. Make sure this script is in the root folder next to 'src'.")
    raise e

# --------------------- 1. Evaluation Dataset Setup ---------------------------- ##
def get_evaluation_prompts():
    """
    مجموعة الأسئلة الطبية الثابتة للاختبار
    """
    eval_set = [
        {"id": 1, "prompt": "What are the primary symptoms of Type 2 Diabetes?", "category": "General"},
        {"id": 2, "prompt": "ما هي الأعراض الجانبية الشائعة لدواء الميتفورمين؟", "category": "Medication (AR)"},
        {"id": 3, "prompt": "How to manage a patient showing acute chest pain before hospital arrival?", "category": "Emergency"},
        {"id": 4, "prompt": "عندي صداع نصفي شديد ومستمر بقاله يومين، أخد علاج إيه؟", "category": "Consultation (EG)"}
    ]
    return eval_set

# --------------------- 2. Evaluation Core Logic ---------------------------- ##
def run_chain_evaluation(chain_instance, searcher_instance, augmentor_instance, prompts, run_name, temperature, model_version):
    """
    دالة تشغيل الأسئلة على الـ Pipeline الكامل وحساب المقاييس
    """
    mlflow.set_experiment('medrag-chainlit-evaluation')
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag('interface', 'Chainlit_Backend')
        mlflow.set_tag('model_version', model_version)
        
        latencies = []
        lengths = []
        
        print(f"\n🚀 Starting Evaluation Run on Chainlit Backend: {run_name} ...")
        
        for item in prompts:
            query = item['prompt']
            start_time = time.time()
            
            try:
                # ── الخطوة 1: Query Augmentation ──
                if asyncio.iscoroutinefunction(augmentor_instance.augment):
                    augmented = asyncio.run(augmentor_instance.augment(query))
                else:
                    augmented = augmentor_instance.augment(query)
                
                # ── الخطوة 2: Hybrid Retrieval ──
                if asyncio.iscoroutinefunction(searcher_instance.search):
                    search_result = asyncio.run(searcher_instance.search(augmented))
                else:
                    search_result = searcher_instance.search(augmented)
                
                # ── الخطوة 3: LLM Generation ──
                if asyncio.iscoroutinefunction(chain_instance.ainvoke):
                    answer = asyncio.run(chain_instance.ainvoke(query, search_result))
                else:
                    answer = chain_instance.ainvoke(query, search_result)
                    
            except Exception as e:
                # خطة بديلة لـ invoke العادية في حالة حدوث أي اختلاف في الـ Async/Sync
                try:
                    answer = chain_instance.invoke(query, search_result)
                except Exception as inner_e:
                    print(f"❌ Failed to run full pipeline for Q{item['id']}: {inner_e}")
                    continue
            
            end_time = time.time()
            
            latency = end_time - start_time
            resp_length = len(str(answer))
            
            latencies.append(latency)
            lengths.append(resp_length)
            
            print(f"🔹 Q{item['id']} [{item['category']}] -> Latency: {latency:.2f}s | Length: {resp_length} chars")
            
            # ── [تعديل جذري آمن]: تسجيل السؤال والرد كـ Tags لضمان عدم حدوث خطأ مسارات ──
            clean_answer = str(answer).replace("\n", " ")
            mlflow.set_tag(f"Q{item['id']}_{item['category']}", f"Prompt: {query} | Response: {clean_answer[:3500]}")
            
        # التأكد من أن هناك بيانات تم تسجيلها بنجاح لمنع الـ ValueError
        if len(latencies) == 0:
            print("❌ Critical: All prompt runs failed. No metrics to log to MLflow.")
            return

        # --------------------- 3. Logging Results to MLflow ---------------------------- ##
        mlflow.log_params({
            'temperature': temperature,
            'total_questions': len(prompts),
            'vector_db': 'Qdrant_Local_Hybrid'
        })
        
        mlflow.log_metrics({
            'avg_latency_seconds': float(np.mean(latencies)),
            'avg_response_length': float(np.mean(lengths)),
            'max_latency': float(np.max(latencies))
        })
        
        print(f"✅ Run '{run_name}' successfully logged to MLflow UI under 'medrag-chainlit-evaluation'!")

# --------------------- 4. Main Orchestrator ---------------------------- ##
def main(temperature: float, model_version: str):
    prompts = get_evaluation_prompts()
    
    print("⏳ Initializing Full MedRAG Pipeline Services (Augmentor, Searcher, Chain)...")
    chain = MedRAGChain()
    searcher = HybridSearcher()
    augmentor = QueryAugmentor()
    
    run_chain_evaluation(
        chain_instance=chain,
        searcher_instance=searcher,
        augmentor_instance=augmentor,
        prompts=prompts, 
        run_name=f'chainlit-medrag-{model_version}', 
        temperature=temperature, 
        model_version=model_version
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedRAG Chainlit External Evaluation Pipeline")
    parser.add_argument('--temperature', '-t', type=float, default=0.2)
    parser.add_argument('--model_version', '-v', type=str, default="v1.0.0")
    args = parser.parse_args()
    
    main(temperature=args.temperature, model_version=args.model_version)