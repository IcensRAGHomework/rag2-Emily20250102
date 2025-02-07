from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)

q1_pdf = "OpenSourceLicenses.pdf"
q2_pdf = "勞動基準法.pdf"


def hw02_1(q1_pdf):
    loader = PyPDFLoader(q1_pdf)
    documents = loader.load()

    page_splitter = CharacterTextSplitter()
    
    # chunks = []
    # for document in documents:
    #      # Here, document.text contains the text for each page
    #     page_chunks = text_splitter.split_text(document.page_content)
    #     chunks.extend(page_chunks)
    # print(f"Total chunks: {len(chunks)}")
    # print(f"Sample chunk: {chunks[0]}")
    # lastChunk = chunks[-1]
    # return lastChunk
    
    page_chunks = page_splitter.split_documents(documents)

    # print(f"Total chunks: {len(page_chunks)}")
    # print(f"Sample chunk: {page_chunks[0]}")
    lastChunk = page_chunks[-1]
    return lastChunk

def hw02_2(q2_pdf):
    loader = PyPDFLoader(q2_pdf)
    documents = loader.load()

    full_text = ""
    for document in documents:
        # full_text += document.page_content
        full_text = full_text + document.page_content +"\n"


    chapter_splitter = RecursiveCharacterTextSplitter(  
        chunk_size = 500,
        chunk_overlap= 0,
        separators=[r"(?:\n|^|\s*)第\s+(?:[一二三四五六七八九十百千萬]+|零)\s+章.*\n"],
        is_separator_regex=True,
        add_start_index = False,
    )
    chapter_chunks = chapter_splitter.split_text(full_text)

    session_splitter = RecursiveCharacterTextSplitter(  
        chunk_size = 10,
        chunk_overlap= 0,
        separators=[r"(?:\n|^)第\s+[0-9]+(?:-[0-9]+)?\s+條.*\n"],
        is_separator_regex=True,
        add_start_index = False,
    )
    index = 0
    for chapter_content in chapter_chunks:
        session_chunks = session_splitter.split_text(chapter_content)
        session_index = 0
        for session_content in session_chunks:
            print(f"---------------{index+session_index} chunk content: {session_content}")
            session_index = session_index+1
        index = index+len(session_chunks)
    
    return index

if __name__ == "__main__":
    print("****************hw02_1******************")
    last_chunk = hw02_1(q1_pdf)
    print(last_chunk)
    print("\n")
    print("****************hw02_2******************")
    chunk_size = hw02_2(q2_pdf)
    print(chunk_size)