"""rag_chain — the minimal, fast version.

    ingest.py    lightweight loaders (PyPDFLoader/Docx2txtLoader/TextLoader)
                 + RecursiveCharacterTextSplitter
    store.py     FastEmbedEmbeddings (ONNX, no PyTorch) + Qdrant, embedded/local,
                 persisted to disk so re-running the app doesn't re-embed
                 files it's already indexed
    chain.py     the canonical LCEL pattern:

                     retriever | format_docs | prompt | llm | StrOutputParser
"""
