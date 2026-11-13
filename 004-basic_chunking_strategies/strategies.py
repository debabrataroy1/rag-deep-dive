import tiktoken
import re

def fixed_size_chunking(text, chunk_size=500, overlap = 50):
    if isinstance(text, list):
        text = '\n'.join(text)
        
    if not isinstance(text, str):
        return []
    
    chunks = []
    text = text.strip()
    step = chunk_size - overlap
    
    for start in range(0,len(text),step):
        if chunk := text[start:start+chunk_size].strip():
            chunks.append(chunk)
        
    return chunks


def fixed_token_chunking(text, chunk_size=100, overlap = 10, model='cl100k_base'):
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    chunks = []
    steps = chunk_size - overlap
    for start in range(0, len(tokens), steps):
        token_chunks = tokens[start:start+chunk_size]
        if chunk := encoding.decode(token_chunks).strip():
            chunks.append(chunk)
            
    return chunks


def sentence_based_chunking(text, max_tokens=100, model="cl100k_base"):
    encoding = tiktoken.get_encoding(model)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    
    chunks, current_chunk, current_tokens = [], [], 0
    
    for s in sentences:
        s_len = len(encoding.encode(s))
        
        # Force standalone if a single sentence exceeds the max token limit
        if s_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_tokens = [], 0
            chunks.append(s)
            continue
            
        # Roll over to a new chunk if limit is reached
        if current_tokens + s_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_tokens = [s], s_len
        else:
            current_chunk.append(s)
            current_tokens += s_len
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks