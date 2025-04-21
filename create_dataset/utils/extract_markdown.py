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
    
    # === 1차 청크 분할 (이 단계에서 오버랩 적용) ===
    initial_chunks = text_splitter.split_text(full_text)
    print(f"초기 분할: 총 {len(initial_chunks)}개의 청크로 나누었습니다.")
    
    # === 텍스트 오버랩 분석 ===
    # 디버깅용: 처음 몇 개 청크의 오버랩 영역 확인
    if len(initial_chunks) >= 2:
        for i in range(min(3, len(initial_chunks) - 1)):
            chunk1 = initial_chunks[i]
            chunk2 = initial_chunks[i + 1]
            # 오버랩 영역 찾기
            overlap_text = find_overlap(chunk1, chunk2)
            if overlap_text:
                print(f"청크 {i}와 {i+1} 사이의 오버랩 (길이: {len(overlap_text)}): {overlap_text[:50]}...")
    
    # === 표 처리 로직 ===
    # 표가 분할되지 않도록 처리하면서 오버랩 유지
    table_processed_chunks = []
    table_buffer = ""
    in_table = False
    
    for i, chunk in enumerate(initial_chunks):
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
                    
                    # 다음 청크와의 오버랩 유지
                    if i+1 < len(initial_chunks):
                        # 다음 청크에서 오버랩 영역을 찾아 추가
                        next_chunk = initial_chunks[i+1]
                        table_buffer = maintain_overlap(table_buffer, next_chunk, chunk_overlap)
                    
                    table_processed_chunks.append(table_buffer)
                    table_buffer = ""
        else:
            # 표가 아닌 경우 그대로 추가
            # 이전에 처리된 청크와의 오버랩은 이미 RecursiveCharacterTextSplitter에서 처리됨
            table_processed_chunks.append(chunk)
    
    # 남은 버퍼 처리
    if table_buffer:
        table_processed_chunks.append(table_buffer)
    
    print(f"표 처리 후: 총 {len(table_processed_chunks)}개의 청크")
    
    # === 구분선 제거 및 작은 청크 처리 ===
    # "-----"만 있는 청크 필터링 및 작은 청크 병합
    filtered_chunks = []
    small_chunks_indices = []  # 작은 청크의 인덱스 저장
    
    # 1단계: 무의미한 구분선 청크 필터링 및 작은 청크 식별
    for i, chunk in enumerate(table_processed_chunks):
        # 청크 내용 정리
        clean_content = chunk.strip()
        
        # 구분선만 있는 청크 제외
        if clean_content and not re.match(r'^-+\s*$', clean_content) and not re.match(r'^[\s\n-]*$', clean_content):
            filtered_chunks.append(chunk)
            
            # 작은 청크 식별 (나중에 병합 처리)
            if len(clean_content) < 100:
                small_chunks_indices.append(len(filtered_chunks) - 1)
    
    print(f"필터링 후: 총 {len(filtered_chunks)}개의 유효 청크 (작은 청크: {len(small_chunks_indices)}개)")
    
    # 2단계: 작은 청크 병합 (오버랩 유지)
    final_chunks = []
    skip_index = set()  # 이미 병합된 청크의 인덱스를 저장
    
    for i in range(len(filtered_chunks)):
        if i in skip_index:
            continue
            
        current_chunk = filtered_chunks[i]
        
        # 현재 청크가 작은 청크인지 확인
        if i in small_chunks_indices:
            # 병합 가능한 인접 청크 찾기
            if i > 0 and i < len(filtered_chunks) - 1:
                # 이전 청크와 다음 청크 모두 있는 경우
                prev_chunk = filtered_chunks[i-1] if i-1 not in skip_index else None
                next_chunk = filtered_chunks[i+1] if i+1 not in skip_index else None
                
                if prev_chunk and next_chunk:
                    # 이전/다음 청크 중 더 작은 쪽과 병합
                    if len(prev_chunk) <= len(next_chunk) and i-1 not in small_chunks_indices:
                        # 이전 청크가 더 작으면 이전 청크에 현재 청크 병합 (오버랩 유지)
                        merged_content = prev_chunk + "\n\n" + current_chunk
                        
                        # 다음 청크와의 오버랩 유지
                        merged_content = maintain_overlap(merged_content, next_chunk, chunk_overlap)
                        
                        # 이미 추가된 이전 청크를 업데이트
                        if final_chunks:  # 안전 검사
                            final_chunks[-1] = merged_content
                        else:
                            final_chunks.append(merged_content)
                        skip_index.add(i)
                    elif i+1 not in small_chunks_indices:
                        # 다음 청크와 병합
                        merged_content = current_chunk + "\n\n" + next_chunk
                        
                        # 병합된 청크의 다음 청크와 오버랩 유지 (있는 경우)
                        if i+2 < len(filtered_chunks):
                            merged_content = maintain_overlap(merged_content, filtered_chunks[i+2], chunk_overlap)
                            
                        final_chunks.append(merged_content)
                        skip_index.add(i+1)
                    else:
                        # 둘 다 작은 청크인 경우 그냥 추가
                        final_chunks.append(current_chunk)
                elif prev_chunk and i-1 not in small_chunks_indices:
                    # 이전 청크만 있고 작은 청크가 아닌 경우
                    merged_content = prev_chunk + "\n\n" + current_chunk
                    # 이미 추가된 이전 청크를 업데이트
                    if final_chunks:  # 안전 검사
                        final_chunks[-1] = merged_content
                    else:
                        final_chunks.append(merged_content)
                    skip_index.add(i)
                elif next_chunk and i+1 not in small_chunks_indices:
                    # 다음 청크만 있고 작은 청크가 아닌 경우
                    merged_content = current_chunk + "\n\n" + next_chunk
                    
                    # 병합된 청크의 다음 청크와 오버랩 유지 (있는 경우)
                    if i+2 < len(filtered_chunks):
                        merged_content = maintain_overlap(merged_content, filtered_chunks[i+2], chunk_overlap)
                        
                    final_chunks.append(merged_content)
                    skip_index.add(i+1)
                else:
                    # 병합할 수 있는 청크가 없는 경우
                    final_chunks.append(current_chunk)
            elif i > 0 and i-1 not in skip_index and i-1 not in small_chunks_indices:
                # 이전 청크만 있는 경우
                merged_content = filtered_chunks[i-1] + "\n\n" + current_chunk
                # 이미 추가된 이전 청크를 업데이트
                if final_chunks:  # 안전 검사
                    final_chunks[-1] = merged_content
                else:
                    final_chunks.append(merged_content)
                skip_index.add(i)
            elif i < len(filtered_chunks) - 1 and i+1 not in skip_index and i+1 not in small_chunks_indices:
                # 다음 청크만 있는 경우
                merged_content = current_chunk + "\n\n" + filtered_chunks[i+1]
                
                # 병합된 청크의 다음 청크와 오버랩 유지 (있는 경우)
                if i+2 < len(filtered_chunks):
                    merged_content = maintain_overlap(merged_content, filtered_chunks[i+2], chunk_overlap)
                    
                final_chunks.append(merged_content)
                skip_index.add(i+1)
            else:
                # 병합할 수 있는 청크가 없는 경우
                final_chunks.append(current_chunk)
        else:
            # 정상 크기 청크는 그대로 추가
            # 다음 청크와의 오버랩 유지
            if i < len(filtered_chunks) - 1 and i+1 not in skip_index:
                current_chunk = maintain_overlap(current_chunk, filtered_chunks[i+1], chunk_overlap)
                
            final_chunks.append(current_chunk)
    
    print(f"최종 결과: 총 {len(final_chunks)}개의 청크 (병합으로 {len(filtered_chunks) - len(final_chunks)}개 감소)")
    
    # 최종 오버랩 확인 (디버깅용)
    if len(final_chunks) >= 2:
        for i in range(min(3, len(final_chunks) - 1)):
            chunk1 = final_chunks[i]
            chunk2 = final_chunks[i + 1]
            # 오버랩 영역 찾기
            overlap_text = find_overlap(chunk1, chunk2)
            if overlap_text:
                print(f"최종 청크 {i}와 {i+1} 사이의 오버랩 (길이: {len(overlap_text)}): {overlap_text[:50]}...")
            else:
                print(f"경고: 청크 {i}와 {i+1} 사이에 오버랩이 없습니다!")
    
    # 결과 파일로 저장 (옵션)
    if save_files:
        if output_dir is None:
            # 기본 출력 디렉토리 설정
            parent_dir = os.path.dirname(pdf_path) if os.path.isfile(pdf_path) else pdf_path
            output_dir = os.path.join(parent_dir, "markdown_chunks")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 설정
        base_name = os.path.basename(pdf_path).replace(".pdf", "") if os.path.isfile(pdf_path) else "combined"
        
        # 청크 파일 저장
        for i, chunk in enumerate(final_chunks):
            chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1:03d}.md")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)
        
        print(f"총 {len(final_chunks)}개의 청크를 {output_dir} 디렉토리에 저장했습니다.")
    
    return final_chunks

def find_overlap(text1, text2, min_overlap=10):
    """두 텍스트 간의 오버랩 영역을 찾습니다."""
    # 두 텍스트가 모두 있는지 확인
    if not text1 or not text2:
        return ""
        
    # 가능한 최대 오버랩 길이 (text2의 길이를 초과할 수 없음)
    max_overlap = min(len(text1), len(text2))
    
    for overlap_size in range(max_overlap, min_overlap-1, -1):
        # text1의 끝부분
        text1_suffix = text1[-overlap_size:]
        # text2의 시작부분
        text2_prefix = text2[:overlap_size]
        
        # 오버랩 확인
        if text1_suffix == text2_prefix:
            return text1_suffix
    
    return ""

def maintain_overlap(current_chunk, next_chunk, overlap_size):
    """현재 청크에 다음 청크와의 오버랩을 유지합니다."""
    # 다음 청크가 충분히 길지 않으면 그대로 반환
    if len(next_chunk) < overlap_size:
        return current_chunk
        
    # 다음 청크의 시작 부분에서 오버랩 크기만큼 가져오기
    overlap_text = next_chunk[:overlap_size]
    
    # 현재 청크에 이미 오버랩 텍스트가 포함되어 있는지 확인
    if current_chunk.endswith(overlap_text):
        return current_chunk
        
    # 오버랩 텍스트의 일부가 이미 현재 청크에 포함되어 있는지 확인
    existing_overlap = find_overlap(current_chunk, next_chunk)
    if existing_overlap:
        # 이미 존재하는 오버랩이 있으면 추가 오버랩 필요 없음
        return current_chunk
        
    # 오버랩 텍스트를 현재 청크 끝에 추가
    # 자연스러운 연결을 위해 줄바꿈 추가
    return current_chunk + "\n\n" + overlap_text

# 사용 예시
# chunks = extract_markdown_from_pdf("example.pdf", chunk_size=1000, chunk_overlap=200, save_files=True)