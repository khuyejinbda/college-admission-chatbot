import pymupdf4llm
import os
import re
import ast
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_markdown_from_pdf(pdf_path, output_dir=None, chunk_size=2000, chunk_overlap=200, save_files=False, min_chunk_size=200):
    """
    pymupdf4llm 라이브러리를 사용해 PDF 문서를 마크다운으로 변환하고 텍스트/표/이미지 등의 요소를 포함한 형태로 추출한 후
    의미론적 단위로 청크를 분할합니다. 표와 텍스트를 동일 청크에 통합 저장합니다.
    페이지 경계에서 끊긴 문장을 연결하는 기능이 강화되었습니다.
    길이가 min_chunk_size 이하인 청크는 인접한 청크와 병합됩니다.
    
    Args:
        pdf_path: PDF 파일 경로 또는 PDF 파일이 있는 디렉토리 경로
        output_dir: 결과를 저장할 디렉토리 경로 (기본값: pdf_path의 상위 디렉토리/markdown_chunks)
        chunk_size: 각 청크의 최대 문자 수 (기본값: 2000)
        chunk_overlap: 청크 간 겹치는 문자 수 (기본값: 200)
        save_files: 결과를 파일로 저장할지 여부 (기본값: False)
        min_chunk_size: 청크의 최소 문자 수, 이보다 작을 경우 병합 (기본값: 200)
    
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
    
    # 각 페이지 정보를 유지하기 위한 파라미터 추가
    docs = reader.load_data(pdf_path, keep_page_info=True)
    
    # 마크다운 문서 처리 및 구조 개선
    processed_docs = []
    page_contents = {}  # 페이지별 내용 저장
    last_page_number = -1
    
    for doc in docs:
        # 페이지 번호 추출 (metadata에서 가져오기)
        page_number = doc.metadata.get('page', last_page_number + 1)
        last_page_number = page_number
        
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
        
        # 페이지별 내용 저장
        if page_number not in page_contents:
            page_contents[page_number] = []
        
        page_contents[page_number].append(text)
        processed_docs.append(text)
    
    # === 페이지 단위로 내용 처리 ===
    page_texts = {}
    
    for page_num, texts in sorted(page_contents.items()):
        # 페이지 내 모든 텍스트 합치기
        page_text = "\n\n".join(texts)
        
        # 페이지 내 텍스트 처리 (문장 연결, 구조 개선 등)
        # 1. 마침표, 물음표, 느낌표 등으로 끝나지 않은 줄을 다음 줄과 연결
        page_text = re.sub(r'([^.!?"\'\]\)\}])\n+((?![#\|\-\*])[가-힣a-zA-Z0-9(])', r'\1 \2', page_text)
        
        # 2. 마지막 단어가 잘렸을 가능성이 있는 경우 처리 (한글 또는 영문자로 끝나는 줄)
        page_text = re.sub(r'([가-힣a-zA-Z])\n+((?![#\|\-\*])[가-힣a-zA-Z0-9])', r'\1\2', page_text)
        
        # 3. 특수한 경우 처리
        page_text = re.sub(r'(보장)\s*\n+\s*(지도)', r'\1 \2', page_text)
        
        # 4. 일반적인 한글 명사로 끝나는 경우와 다음 줄의 시작이 조사일 경우 연결
        page_text = re.sub(r'([가-힣]+)\n+([은는이가을를에의로])', r'\1\2', page_text)
        
        # 5. 과도한 줄바꿈 정리 (3개 이상 -> 2개로)
        page_text = re.sub(r'\n{3,}', r'\n\n', page_text)
        
        # 6. 표와 주변 텍스트의 관계를 명확히 하기 위한 처리
        page_text = re.sub(r'([^\n])\n(\|[-\s|]+\|)', r'\1\n\n\2', page_text)
        page_text = re.sub(r'(\|\s*\n)(?!\|)([^\n])', r'\1\n\2', page_text)
        
        # 7. 구분선(----) 제거
        page_text = re.sub(r'\n\s*-{3,}\s*\n', r'\n\n', page_text)
        page_text = re.sub(r'^\s*-{3,}\s*$', r'', page_text, flags=re.MULTILINE)
        
        # 8-10. 기타 문서 구조 개선
        page_text = re.sub(r'\n#\s*\n', r'\n\n', page_text)
        page_text = re.sub(r'(?<!\n)#{1,6}\s+', r'\n\n\g<0>', page_text)
        page_text = re.sub(r'\.\n([가-힣a-zA-Z0-9])', r'. \1', page_text)
        
        page_texts[page_num] = page_text
    
    # === 표 감지 로직 ===
    # 각 페이지별 표 위치 식별
    page_tables = {}
    
    for page_num, page_text in page_texts.items():
        tables = []
        table_pattern = r'(\|[^\n]*\|(\n\|[^\n]*\|)+)'
        
        for match in re.finditer(table_pattern, page_text):
            tables.append((match.start(), match.end(), match.group(0)))
        
        page_tables[page_num] = tables
        
        if tables:
            print(f"페이지 {page_num}에서 {len(tables)}개의 표를 감지했습니다.")
    
    # === 페이지 간 표 연속성 처리 ===
    # 페이지 간에 연속된 표 식별
    connected_pages = []
    
    for page_num in sorted(page_texts.keys()):
        if page_num + 1 in page_texts:
            current_page_text = page_texts[page_num]
            next_page_text = page_texts[page_num + 1]
            
            # 현재 페이지가 표로 끝나고 다음 페이지가 표로 시작하는지 확인
            if (current_page_text.rstrip().endswith('|') and 
                '|' in next_page_text.lstrip()[:50]):
                connected_pages.append((page_num, page_num + 1))
                print(f"페이지 {page_num}과 {page_num + 1} 사이에 연속된 표를 감지했습니다.")
    
    # === 페이지 그룹화 ===
    # 연결된 페이지들을 그룹화
    page_groups = []
    processed = set()
    
    # 먼저 연결된 페이지 그룹 만들기
    for page1, page2 in connected_pages:
        if page1 not in processed and page2 not in processed:
            group = {page1, page2}
            processed.add(page1)
            processed.add(page2)
            
            # 그룹에 추가적인 연결된 페이지가 있는지 확인
            changed = True
            while changed:
                changed = False
                for p1, p2 in connected_pages:
                    if p1 in group and p2 not in group:
                        group.add(p2)
                        processed.add(p2)
                        changed = True
                    elif p2 in group and p1 not in group:
                        group.add(p1)
                        processed.add(p1)
                        changed = True
            
            page_groups.append(sorted(group))
    
    # 그룹화되지 않은 페이지 추가
    for page_num in sorted(page_texts.keys()):
        if page_num not in processed:
            page_groups.append([page_num])
    
    # 페이지 그룹 정렬
    page_groups.sort(key=lambda x: x[0])
    
    # === 페이지 그룹별 청크 생성 ===
    initial_chunks = []
    
    for group in page_groups:
        # 구분선(-----)만 있는 페이지는 건너뛰기
        valid_pages = []
        for page in group:
            page_content = page_texts[page].strip()
            # 구분선만 있는지 확인
            if page_content and not re.match(r'^-{3,}\s*$', page_content) and not re.match(r'^[\s\n-]*$', page_content):
                valid_pages.append(page)
        
        # 유효한 페이지가 없는 경우 건너뛰기
        if not valid_pages:
            continue
            
        group_text = "\n\n".join([page_texts[page] for page in valid_pages])
        
        # 구분선 제거 전처리
        group_text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', group_text)
        group_text = re.sub(r'^\s*-{3,}\s*$', '', group_text, flags=re.MULTILINE)
        
        # 표 보호 처리
        protected_text = protect_table_format(group_text)
        
        # 청크 크기를 고려한 분할 여부 결정
        if len(protected_text) <= chunk_size:
            # 청크 크기 이하면 그대로 유지
            clean_text = protected_text.replace("<TABLE_START>", "").replace("<TABLE_END>", "")
            # 최종 정리 - 빈 줄 제거 및 구분선 제거
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
            clean_text = re.sub(r'^\s*-{3,}\s*$', '', clean_text, flags=re.MULTILINE)
            
            if clean_text.strip():  # 내용이 있는 경우만 추가
                initial_chunks.append(clean_text)
        else:
            # 청크 크기 초과 시 분할 필요
            # 표 보존하면서 분할
            splits = split_with_table_preservation(protected_text, chunk_size, chunk_overlap)
            
            # 각 분할된 청크에서 구분선 제거 최종 정리
            for i, split in enumerate(splits):
                clean_split = re.sub(r'^\s*-{3,}\s*$', '', split, flags=re.MULTILINE)
                clean_split = re.sub(r'\n{3,}', '\n\n', clean_split)
                
                if clean_split.strip():  # 내용이 있는 경우만 추가
                    initial_chunks.append(clean_split)
    
    # === 청크 간 오버랩 생성 ===
    # 인접한 청크 간 오버랩 생성
    final_chunks = []
    
    for i in range(len(initial_chunks)):
        current_chunk = initial_chunks[i]
        
        # 첫 번째 청크는 그대로 추가
        if i == 0:
            final_chunks.append(current_chunk)
            continue
        
        # 이전 청크와 현재 청크 사이의 오버랩 생성
        prev_chunk = initial_chunks[i-1]
        
        # 오버랩이 있는지 확인
        existing_overlap = find_overlap(prev_chunk, current_chunk)
        
        # 오버랩이 없거나 충분하지 않은 경우, 이전 청크의 마지막 부분을 현재 청크 앞에 추가
        if not existing_overlap or len(existing_overlap) < chunk_overlap:
            # 이전 청크의 마지막 부분 가져오기
            if len(prev_chunk) > chunk_overlap:
                overlap_from_prev = prev_chunk[-chunk_overlap:]
                
                # 문장 시작 찾기 (자연스러운 중첩을 위해)
                sentence_start = find_sentence_start(overlap_from_prev)
                if sentence_start > 0:
                    overlap_from_prev = overlap_from_prev[sentence_start:]
                
                # 오버랩 적용
                if not current_chunk.startswith(overlap_from_prev):
                    # 표가 포함된 경우 문제가 발생할 수 있으므로 표가 있는지 확인
                    if not '|' in overlap_from_prev and not '<TABLE_START>' in current_chunk[:100]:
                        current_chunk = overlap_from_prev + current_chunk
        
        final_chunks.append(current_chunk)
    
    # === 작은 청크 처리 (새로 추가된 기능) ===
    merged_chunks = merge_small_chunks(final_chunks, min_chunk_size)
    
    print(f"최종 결과: 총 {len(merged_chunks)}개의 청크를 생성했습니다. (작은 청크 병합 후)")
    
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
        for i, chunk in enumerate(merged_chunks):
            chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1:03d}.md")
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk)
        
        print(f"총 {len(merged_chunks)}개의 청크를 {output_dir} 디렉토리에 저장했습니다.")
    
    return merged_chunks

def merge_small_chunks(chunks, min_chunk_size):
    """
    길이가 min_chunk_size보다 작은 청크들을 인접한 청크와 병합합니다.
    병합 시 이전이나 이후 청크 중 더 작은 크기의 청크와 병합합니다.
    
    Args:
        chunks: 병합할 청크 리스트
        min_chunk_size: 최소 청크 크기 (이보다 작은 청크는 병합됨)
    
    Returns:
        병합된 청크 리스트
    """
    if not chunks:
        return []
    
    # 병합이 필요한 청크가 있는지 확인
    need_merge = any(len(chunk) < min_chunk_size for chunk in chunks)
    if not need_merge:
        return chunks
    
    result = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        
        # 현재 청크가 최소 크기보다 작은 경우 처리
        if len(current_chunk) < min_chunk_size:
            # 첫 번째 청크인 경우 (뒤의 청크와만 병합 가능)
            if i == 0:
                if i + 1 < len(chunks):  # 다음 청크가 있는 경우
                    merged_chunk = current_chunk + "\n\n" + chunks[i + 1]
                    result.append(merged_chunk)
                    i += 2  # 다음 청크도 처리했으므로 +2
                else:  # 마지막 청크인 경우 그대로 추가
                    result.append(current_chunk)
                    i += 1
            # 마지막 청크인 경우 (앞의 청크와만 병합 가능)
            elif i == len(chunks) - 1:
                # 이미 앞 청크는 처리되었으므로, 결과에서 마지막 항목을 빼고 병합
                prev_chunk = result.pop()
                merged_chunk = prev_chunk + "\n\n" + current_chunk
                result.append(merged_chunk)
                i += 1
            # 중간 청크인 경우 (앞뒤 청크 비교 후 더 작은 쪽과 병합)
            else:
                prev_chunk = chunks[i - 1] if i - 1 >= 0 else ""
                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else ""
                
                # 앞뒤 청크 길이 비교 (이미 앞 청크가 result에 있으면 복잡해지므로 길이만 비교)
                prev_chunk_size = len(prev_chunk)
                next_chunk_size = len(next_chunk)
                
                # 더 작은 쪽과 병합 (같으면 앞쪽과 병합)
                if not next_chunk or (prev_chunk and prev_chunk_size <= next_chunk_size):
                    # 앞의 청크와 병합 (앞 청크는 이미 result에 추가되어 있음)
                    prev_chunk = result.pop()  # 결과에서 마지막 항목 제거
                    merged_chunk = prev_chunk + "\n\n" + current_chunk
                    result.append(merged_chunk)
                    i += 1
                else:
                    # 뒤의 청크와 병합
                    merged_chunk = current_chunk + "\n\n" + next_chunk
                    result.append(merged_chunk)
                    i += 2  # 다음 청크도 처리했으므로 +2
        else:
            # 현재 청크 크기가 충분한 경우 그대로 추가
            result.append(current_chunk)
            i += 1
    
    # 병합 후에도 작은 청크가 있는지 재확인
    if any(len(chunk) < min_chunk_size for chunk in result):
        # 재귀적으로 다시 병합 수행 (최대 3회까지)
        return merge_small_chunks(result, min_chunk_size)
    
    return result

def protect_table_format(text):
    """표 형식을 보존하기 위해 특수 토큰으로 래핑합니다. 구분선은 무시합니다."""
    lines = text.split('\n')
    in_table = False
    protected_lines = []
    
    for i, line in enumerate(lines):
        # 구분선 무시 (-----와 같은 패턴)
        if re.match(r'^\s*-{3,}\s*$', line):
            continue
            
        # 표 시작 감지
        if re.match(r'^\s*\|', line) and not in_table:
            in_table = True
            protected_lines.append("<TABLE_START>")
            protected_lines.append(line)
        # 표 내부 라인
        elif in_table and re.match(r'^\s*\|', line):
            protected_lines.append(line)
        # 표 종료 감지
        elif in_table and not re.match(r'^\s*\|', line):
            in_table = False
            protected_lines.append("<TABLE_END>")
            
            # 종료 후 라인이 구분선이 아닌 경우만 추가
            if not re.match(r'^\s*-{3,}\s*$', line):
                protected_lines.append(line)
        # 일반 텍스트
        else:
            protected_lines.append(line)
    
    # 열린 표가 있으면 닫기
    if in_table:
        protected_lines.append("<TABLE_END>")
        
    return '\n'.join(protected_lines)

def split_with_table_preservation(text, chunk_size, chunk_overlap):
    """표를 보존하면서 텍스트를 청크로 분할합니다. 구분선은 무시합니다."""
    chunks = []
    
    # 구분선 제거 (전처리)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    
    # 특수 토큰으로 분할
    table_sections = re.split(r'(<TABLE_START>.*?<TABLE_END>)', text, flags=re.DOTALL)
    
    current_chunk = ""
    last_chunk_end = ""  # 마지막 청크의 끝부분 (오버랩용)
    
    for section in table_sections:
        # 구분선만 있는 섹션 건너뛰기
        if re.match(r'^\s*-{3,}\s*$', section.strip()):
            continue
            
        if section.startswith("<TABLE_START>") and section.endswith("<TABLE_END>"):
            # 표 섹션 처리
            table_content = section.replace("<TABLE_START>", "").replace("<TABLE_END>", "")
            
            # 표 내용이 비어있거나 구분선만 있는 경우 건너뛰기
            if not table_content.strip() or re.match(r'^\s*-{3,}\s*$', table_content.strip()):
                continue
                
            # 현재 청크에 표를 추가했을 때 최대 크기를 초과하는지 확인
            if len(current_chunk) + len(table_content) > chunk_size and current_chunk:
                # 현재 청크 저장
                chunks.append(current_chunk)
                
                # 오버랩을 위해 마지막 청크 끝부분 저장
                if len(current_chunk) > chunk_overlap:
                    last_chunk_end = current_chunk[-chunk_overlap:]
                else:
                    last_chunk_end = current_chunk
                
                # 새 청크 시작 - 이전 청크와의 오버랩 추가
                if last_chunk_end and not table_content.startswith(last_chunk_end):
                    current_chunk = table_content
                else:
                    current_chunk = table_content
            else:
                # 현재 청크에 표 추가
                if current_chunk and not current_chunk.endswith("\n\n"):
                    current_chunk += "\n\n"
                current_chunk += table_content
        else:
            # 일반 텍스트 섹션 처리
            # 구분선 패턴 제거
            clean_section = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', section)
            clean_section = re.sub(r'^\s*-{3,}\s*$', '', clean_section, flags=re.MULTILINE)
            
            if not clean_section.strip():
                # 빈 섹션은 건너뛰기
                continue
                
            # 텍스트 분할기 설정 - 오버랩 적용
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n"]
            )
            
            # 현재 청크가 비어있지 않고 텍스트를 추가하면 크기를 초과하는 경우
            if current_chunk and len(current_chunk) + len(clean_section) > chunk_size:
                # 현재 청크 저장
                chunks.append(current_chunk)
                
                # 오버랩을 위해 마지막 청크 끝부분 저장
                if len(current_chunk) > chunk_overlap:
                    last_chunk_end = current_chunk[-chunk_overlap:]
                else:
                    last_chunk_end = current_chunk
                
                # 새 청크 시작 - 텍스트 분할 적용
                text_chunks = text_splitter.split_text(clean_section)
                
                # 첫 번째 분할 청크는 현재 청크로 설정하고 나머지는 개별 청크로 추가
                if text_chunks:
                    # 첫 번째 청크에 오버랩 추가
                    first_chunk = text_chunks[0]
                    # 이미 오버랩이 있는지 확인
                    if not last_chunk_end or first_chunk.startswith(last_chunk_end):
                        current_chunk = first_chunk
                    else:
                        # 자연스러운 문장 시작점 찾기
                        sentence_start = find_sentence_start(last_chunk_end)
                        if sentence_start > 0:
                            overlap_text = last_chunk_end[sentence_start:]
                            if not first_chunk.startswith(overlap_text):
                                current_chunk = overlap_text + first_chunk
                            else:
                                current_chunk = first_chunk
                        else:
                            current_chunk = first_chunk
                    
                    # 나머지 청크들 간에도 오버랩 적용
                    for i in range(1, len(text_chunks)):
                        chunks.append(current_chunk)
                        prev_chunk = current_chunk
                        current_chunk = text_chunks[i]
                        
                        # 인접한 청크 간 오버랩 확인
                        if not current_chunk.startswith(prev_chunk[-chunk_overlap:]) and len(prev_chunk) > chunk_overlap:
                            overlap_text = prev_chunk[-chunk_overlap:]
                            sentence_start = find_sentence_start(overlap_text)
                            if sentence_start > 0:
                                overlap_text = overlap_text[sentence_start:]
                                if not current_chunk.startswith(overlap_text):
                                    current_chunk = overlap_text + current_chunk
                else:
                    current_chunk = ""
            else:
                # 현재 청크에 텍스트 추가
                if current_chunk and not current_chunk.endswith("\n\n") and not clean_section.startswith("\n"):
                    current_chunk += "\n\n"
                current_chunk += clean_section
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    # 특수 토큰 제거 (남아있을 경우)
    for i in range(len(chunks)):
        chunks[i] = chunks[i].replace("<TABLE_START>", "").replace("<TABLE_END>", "")
    
    return chunks

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

def find_sentence_start(text):
    """텍스트 내에서 자연스러운 문장 시작점을 찾습니다."""
    # 마침표, 물음표, 느낌표 등 문장 종결 부호 위치 찾기
    sentence_ends = [m.start() for m in re.finditer(r'[.!?]\s+', text)]
    
    if sentence_ends:
        # 가장 마지막 문장 종결 부호 다음 위치 반환
        last_sentence_end = sentence_ends[-1]
        return last_sentence_end + 2  # 종결 부호와 공백 건너뛰기
    
    # 문장 종결 부호가 없으면 단락 구분자 찾기
    paragraph_breaks = [m.start() for m in re.finditer(r'\n\n', text)]
    if paragraph_breaks:
        # 가장 마지막 단락 구분자 다음 위치 반환
        last_paragraph_break = paragraph_breaks[-1]
        return last_paragraph_break + 2
    
    return 0  # 적절한 시작점을 찾지 못한 경우