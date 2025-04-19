import pymupdf4llm
import os
import re
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_markdown_from_pdf(pdf_path, output_dir=None, chunk_size=1000, chunk_overlap=200, save_files=False):
    """
    pymupdf4llm 라이브러리를 사용해 PDF 문서를 마크다운으로 변환하고 텍스트/표/이미지 등의 요소를 포함한 형태로 추출한 후
    의미론적 단위로 청크를 분할합니다. 표와 텍스트를 동일 청크에 통합 저장합니다.
    페이지 경계에서 끊긴 문장을 연결하는 기능이 강화되었습니다.
    
    Args:
        pdf_path: PDF 파일 경로 또는 PDF 파일이 있는 디렉토리 경로
        output_dir: 결과를 저장할 디렉토리 경로 (기본값: pdf_path의 상위 디렉토리/markdown_chunks)
        chunk_size: 각 청크의 최대 문자 수 (기본값: 1000)
        chunk_overlap: 청크 간 겹치는 문자 수 (기본값: 200)
        save_files: 결과를 파일로 저장할지 여부 (기본값: False)
    
    Returns:
        단일 파일 처리 시: 마크다운 형식의 텍스트 청크 리스트
        디렉토리 처리 시: {파일명: 청크 리스트} 형태의 딕셔너리
    """
    
    # PDF 파일이 존재하는지 확인
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    
    print(f"PDF 파일 로드 중: {pdf_path}")
    
    # pymupdf4llm의 LlamaMarkdownReader를 사용하여 PDF 파일을 마크다운으로 변환
    reader = pymupdf4llm.LlamaMarkdownReader()
    
    docs = reader.load_data(pdf_path)
    
    # 마크다운 문서 처리 및 구조 개선
    processed_docs = []
    for doc in docs:
        # 문서 구조를 강화하기 위한 후처리
        text = doc.text
        
        # 배열 형식으로 반환된 경우 처리
        if text.startswith("['") and text.endswith("']"):
            try:
                # 문자열을 리스트로 파싱
                text_list = ast.literal_eval(text)
                # 리스트를 적절하게 조합
                text = "\n\n".join(text_list)
            except (SyntaxError, ValueError):
                # 배열 형식이 아니거나 파싱 오류 시 원본 사용
                pass
        
        # 특수 문자 정리 (불릿 포인트 등 변환)
        text = text.replace('\uf09f', '• ')
        
        processed_docs.append(text)
    
    # 모든 문서 텍스트 합치기
    full_text = "\n\n".join(processed_docs)
    
    # === 페이지 경계에서 끊긴 문장 연결 강화 ===
    
    # 1. 마침표, 물음표, 느낌표 등으로 끝나지 않은 줄을 다음 줄과 연결
    # 단, 제목('#'으로 시작), 표('|'로 시작), 구분선('---'로 시작) 등은 제외
    full_text = re.sub(r'([^.!?"\'\]\)\}])\n+((?![#\|\-\*])[가-힣a-zA-Z0-9(])', r'\1 \2', full_text)
    
    # 2. 마지막 단어가 잘렸을 가능성이 있는 경우 처리 (한글 또는 영문자로 끝나는 줄)
    full_text = re.sub(r'([가-힣a-zA-Z])\n+((?![#\|\-\*])[가-힣a-zA-Z0-9])', r'\1\2', full_text)
    
    # 3. 특수한 경우: "보장" 또는 "보장 "으로 끝나는 문장과 "지도"로 시작하는 문장 연결
    # 예: "최소 성취 수준 보장" + "지도는..." -> "최소 성취 수준 보장 지도는..."
    full_text = re.sub(r'(보장)\s*\n+\s*(지도)', r'\1 \2', full_text)
    
    # 4. 일반적인 한글 명사로 끝나는 경우와 다음 줄의 시작이 조사일 경우 연결
    # 예: "학생" + "에게는..." -> "학생에게는..."
    full_text = re.sub(r'([가-힣]+)\n+([은는이가을를에의로])', r'\1\2', full_text)
    
    # 5. 과도한 줄바꿈 정리 (3개 이상 -> 2개로)
    full_text = re.sub(r'\n{3,}', r'\n\n', full_text)
    
    # 6. 표와 주변 텍스트의 관계를 명확히 하기 위한 처리
    # 표 시작 전에 최소 한 줄의 공백 추가
    full_text = re.sub(r'([^\n])\n(\|[-\s|]+\|)', r'\1\n\n\2', full_text)
    
    # 표 끝 후에 최소 한 줄의 공백 추가
    full_text = re.sub(r'(\|\s*\n)(?!\|)([^\n])', r'\1\n\2', full_text)
    
    # 7. 빈 헤더(#) 정리
    full_text = re.sub(r'\n#\s*\n', r'\n\n', full_text)
    
    # 8. 섹션 구분을 위한 패턴 강화 (제목 형식 강화)
    full_text = re.sub(r'(?<!\n)#{1,6}\s+', r'\n\n\g<0>', full_text)
    
    # 9. ----- 구분자 정리 (중복 제거 및 표준화)
    full_text = re.sub(r'\n{2,}-----\n{2,}', r'\n\n-----\n\n', full_text)
    
    # 10. ".\n" 패턴 제거 (문장 분할 방지)
    full_text = re.sub(r'\.\n([가-힣a-zA-Z0-9])', r'. \1', full_text)
    
    # 의미론적 단위로 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # 의미론적 분할을 위한 구분자 설정 (우선순위 순)
        separators=[
            "\n# ",              # 1단계 제목 (새 챕터 시작)
            "\n## ",             # 2단계 제목 (새 섹션 시작)
            "\n### ",            # 3단계 제목 
            "\n#### ",           # 4단계 제목
            "\n\n"               # 단락 구분 (문단 간 분할 우선)
        ]
    )
    
    # 표를 포함한 청크로 분할
    chunks = text_splitter.split_text(full_text)
    
    # 표가 분할되지 않도록 후처리
    processed_chunks = []
    table_buffer = ""
    in_table = False
    
    for chunk in chunks:
        # 표 시작 패턴 확인 (표 시작 라인이 있고 표 끝 패턴이 없는 경우)
        if not in_table and re.search(r'\n\|[-\s|]+\|', chunk) and not chunk.count('|') % 2 == 0:
            in_table = True
            table_buffer = chunk
        # 표 내부 또는 표 끝 패턴 확인
        elif in_table:
            table_buffer += "\n" + chunk
            # 각 행의 열 수가 일치하는지 확인 (표가 완성되었는지 여부)
            table_lines = [line for line in table_buffer.split('\n') if line.startswith('|') and line.endswith('|')]
            if len(table_lines) >= 2:  # 헤더 줄과 구분선 최소 2줄 이상 필요
                pipe_counts = [line.count('|') for line in table_lines]
                if len(set(pipe_counts)) == 1:  # 모든 행의 열 수가 동일하면 표가 완성됨
                    in_table = False
                    processed_chunks.append(table_buffer)
                    table_buffer = ""
        else:
            processed_chunks.append(chunk)
    
    # 남은 버퍼 처리
    if table_buffer:
        processed_chunks.append(table_buffer)
    
    # === 청크 간 문장 연결 후처리 ===
    connected_chunks = []
    prev_chunk = ""
    
    for i, chunk in enumerate(processed_chunks):
        clean_chunk = chunk.strip()
        
        # 첫 번째 청크가 아니고, 이전 청크가 완전한 문장으로 끝나지 않은 경우
        if i > 0 and not re.search(r'[.!?]["\'\)\]]*\s*$', prev_chunk):
            # 현재 청크가 소문자나 한글로 시작하는 경우 (제목이나 새로운 섹션이 아닌 경우)
            if re.match(r'^[a-z가-힣]', clean_chunk) and not re.match(r'^[#\|\-\*]', clean_chunk):
                # 이전 청크의 마지막 단어와 현재 청크의 첫 단어 사이에 특별한 관계가 있는지 확인
                prev_words = prev_chunk.split()
                if prev_words and len(prev_words[-1]) >= 1:
                    # 이전 청크의 마지막 단어와 현재 청크를 결합
                    connected_chunks[-1] = connected_chunks[-1] + " " + clean_chunk
                    continue
        
        connected_chunks.append(clean_chunk)
        prev_chunk = clean_chunk
    
    print(f"총 {len(connected_chunks)}개의 청크로 나누었습니다.")
    
    # "-----"만 있는 청크 필터링
    # "-----"만 있는 청크 필터링 (기존 코드와 동일)
    filtered_chunks = []
    for chunk in connected_chunks:
        # "-----"만 있는 청크 또는 "-----"가 대부분인 청크 제외
        # 공백과 줄바꿈을 제거한 후 확인
        clean_content = chunk.strip()
        if clean_content and not re.match(r'^-+\s*$', clean_content):
            # 추가 검사: 내용이 구분선, 공백, 줄바꿈으로만 구성된 경우도 제외
            if not re.match(r'^[\s\n-]*$', clean_content):
                filtered_chunks.append(chunk)
    
    print(f"총 {len(connected_chunks)}개의 청크 중 {len(filtered_chunks)}개의 유효한 청크를 추출했습니다.")
    
    # === 작은 청크 병합 처리 ===
    merged_chunks = []
    skip_index = set()  # 이미 병합된 청크의 인덱스를 저장
    
    for i in range(len(filtered_chunks)):
        if i in skip_index:
            continue
            
        current_chunk = filtered_chunks[i]
        current_length = len(current_chunk)
        
        # 현재 청크가 최소 크기보다 작은 경우 병합 고려
        if current_length < 100: # 최소 크기 설정
            # 병합할 수 있는 이전/다음 청크 확인
            prev_chunk_idx = i - 1
            next_chunk_idx = i + 1
            
            # 이전 청크와 다음 청크 중 어느 것과 병합할지 결정
            if prev_chunk_idx >= 0 and next_chunk_idx < len(filtered_chunks):
                prev_chunk_len = len(filtered_chunks[prev_chunk_idx])
                next_chunk_len = len(filtered_chunks[next_chunk_idx])
                
                # 이전/다음 청크 중 더 작은 쪽에 병합
                if prev_chunk_len <= next_chunk_len and prev_chunk_idx not in skip_index:
                    # 이전 청크가 더 작으면 이전 청크에 현재 청크 병합
                    merged_content = filtered_chunks[prev_chunk_idx] + "\n\n" + current_chunk
                    # 이미 추가된 이전 청크를 업데이트
                    merged_chunks[-1] = merged_content
                    skip_index.add(i)  # 현재 청크는 건너뛰기 표시
                    print(f"청크 {i}(길이: {current_length})를 이전 청크 {prev_chunk_idx}에 병합했습니다.")
                elif next_chunk_idx not in skip_index:
                    # 다음 청크가 더 작거나 같으면 다음 청크와 병합 후 저장
                    merged_content = current_chunk + "\n\n" + filtered_chunks[next_chunk_idx]
                    merged_chunks.append(merged_content)
                    skip_index.add(next_chunk_idx)  # 다음 청크는 건너뛰기 표시
                    print(f"청크 {i}(길이: {current_length})와 다음 청크 {next_chunk_idx}를 병합했습니다.")
                else:
                    # 둘 다 이미 병합된 경우 그냥 추가
                    merged_chunks.append(current_chunk)
            elif prev_chunk_idx >= 0 and prev_chunk_idx not in skip_index:
                # 다음 청크가 없고 이전 청크만 있는 경우
                merged_content = filtered_chunks[prev_chunk_idx] + "\n\n" + current_chunk
                merged_chunks[-1] = merged_content  # 이미 추가된 이전 청크를 업데이트
                skip_index.add(i)  # 현재 청크는 건너뛰기 표시
                print(f"청크 {i}(길이: {current_length})를 이전 청크 {prev_chunk_idx}에 병합했습니다.")
            elif next_chunk_idx < len(filtered_chunks) and next_chunk_idx not in skip_index:
                # 이전 청크가 없고 다음 청크만 있는 경우
                merged_content = current_chunk + "\n\n" + filtered_chunks[next_chunk_idx]
                merged_chunks.append(merged_content)
                skip_index.add(next_chunk_idx)  # 다음 청크는 건너뛰기 표시
                print(f"청크 {i}(길이: {current_length})와 다음 청크 {next_chunk_idx}를 병합했습니다.")
            else:
                # 병합할 수 있는 청크가 없는 경우
                merged_chunks.append(current_chunk)
        else:
            # 최소 크기 이상인 청크는 그대로 추가
            merged_chunks.append(current_chunk)
    
    print(f"총 {len(filtered_chunks)}개의 청크 중 {len(merged_chunks)}개의 병합된 청크를 생성했습니다.")
    print(f"병합으로 {len(filtered_chunks) - len(merged_chunks)}개의 청크가 줄었습니다.")
    
    return merged_chunks