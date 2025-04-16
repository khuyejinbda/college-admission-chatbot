# 전체 파이프라인 함수
from utils import qa_claude, critique_gpt, final_qa_gpt, extract_markdown

from typing import List, Dict, Any
import pymupdf4llm
import os
import json
import nest_asyncio
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pandas as pd

from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()




nest_asyncio.apply()

def qa_generation_pipeline(elements: List[str], domain: str, num_questions: str = "2") -> List[Dict[str, str]]:
    """
    전체 QA 생성 파이프라인 함수
    
    Parameters:
        elements (List[str]): 텍스트 청크 목록
        domain (str): 도메인 (예: 고교학점제)
        num_questions (str): 각 청크당 생성할 질문 수
    
    Returns:
        List[Dict[str, str]]: 최종 QA 쌍 목록
    """
    final_qa_pairs = []
    
    try:
        # 1. 초기 QA 쌍 생성
        initial_qa_pairs = qa_claude.generate_qa_pairs(elements, domain, num_questions)
        
        # 2. QA 쌍에 대한 비판 생성
        critique_qa_pair = critique_gpt.critique_qa_pairs(initial_qa_pairs, domain)
        
        # 3. 비판을 반영한 최종 QA 쌍 생성
        final_qa_pairs = final_qa_gpt.generate_final_qa_pairs(critique_qa_pair, domain)
    
    except Exception as e:
        print('='*50)
        print(f"[에러 발생] | {e} | {elements[0][:100]}...")  # 청크의 첫 부분만 출력
        print('='*50)
        
    return final_qa_pairs

def process_all_pdfs(pdf_dir='./source_data/pdf/', 
                    output_dir='../output_data/',
                    domain="Default",
                    chunk_size=500,
                    chunk_overlap=100,
                    questions_per_chunk="2"):
    """
    지정된 디렉토리에 있는 모든 PDF 파일을 처리하여 QA 셋을 생성합니다.
    
    Parameters:
        pdf_dir (str): PDF 파일이 저장된 디렉토리 경로
        output_dir (str): 결과를 저장할 디렉토리 경로
        domain (str): QA 생성의 도메인
        chunk_size (int): 청크 크기
        chunk_overlap (int): 청크 간 겹침 크기
        questions_per_chunk (str): 각 청크당 생성할 질문 수
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # PDF 파일 목록 가져오기
    pdf_list = [f"{pdf_dir}/{pdf}" for pdf in os.listdir(pdf_dir) if pdf.endswith('.pdf')]
    print(f"총 {len(pdf_list)}개의 PDF 파일을 찾았습니다.")
    
    # 모든 QA 쌍을 저장할 리스트
    all_qa_pairs = []
    
    # PDF 파일별로 처리
    for pdf_path in tqdm(pdf_list, desc="PDF 처리 중"):
        try:
            # PDF 파일명 추출 (확장자 제외)
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            
            print(f"\n[처리 시작] {pdf_name}")
            
            # PDF에서 마크다운 청크 추출
            chunks = extract_markdown.extract_markdown_from_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # 빈 청크 제거
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            if not chunks:
                print(f"[주의] {pdf_name}에서 유효한 청크를 추출하지 못했습니다.")
                continue
            
            # 도메인 설정 - PDF 파일명을 기본 도메인으로 사용하거나 지정된 도메인 사용
            current_domain = domain if domain != "Default" else pdf_name
            
            # 청크를 작은 배치로 나누어 처리 (API 호출 제한 고려)
            batch_size = 3  # 한 번에 처리할 청크 수
            qa_pairs = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                print(f"배치 처리 중: {i//batch_size + 1}/{(len(chunks)+batch_size-1)//batch_size}")
                
                # QA 셋 생성
                batch_qa_pairs = qa_generation_pipeline(batch_chunks, current_domain, questions_per_chunk)
                qa_pairs.extend(batch_qa_pairs)
            
            # 각 QA 쌍에 PDF 출처 정보 추가
            # for qa_pair in qa_pairs:
            #     qa_pair["SOURCE"] = pdf_name
            
            # 전체 QA 쌍 목록에 추가
            all_qa_pairs.extend(qa_pairs)
            
            # 현재 PDF의 결과를 JSONL으로 저장
            pdf_output_path = f"{output_dir}/{pdf_name}_qa.jsonl"
            with open(pdf_output_path, 'w', encoding='utf-8') as f:
                for qa_pair in qa_pairs:
                    f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            
            print(f"[처리 완료] {pdf_name}: {len(qa_pairs)}개의 QA 쌍 생성")
            
        except Exception as e:
            print(f"[에러] {pdf_path} 처리 중 오류 발생: {str(e)}")
    
    # 모든 결과를 하나의 JSONL 파일로 저장
    all_output_path = f"{output_dir}/all_qa_pairs.jsonl"
    with open(all_output_path, 'w', encoding='utf-8') as f:
        for qa_pair in all_qa_pairs:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
    
    print(f"\n처리 완료! 총 {len(all_qa_pairs)}개의 QA 쌍이 생성되었습니다.")
    print(f"결과는 다음 위치에 저장되었습니다:")
    print(f"- JSONL: {all_output_path}")
    
    return all_qa_pairs