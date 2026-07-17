"""
快速测试脚本

用于验证各个模块是否正常工作。
运行方式：python test_modules.py

注意：需要先在 .env 中配置好 OPENAI_API_KEY
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_loader():
    """测试文档加载模块"""
    print("\n" + "=" * 50)
    print("📄 测试文档加载模块")
    print("=" * 50)

    from ingestion.loader import DocumentLoader

    loader = DocumentLoader()
    docs = loader.load_file("data/sample_docs/ai_basics.md")
    print(f"加载了 {len(docs)} 个文档")
    print(f"文档长度: {len(docs[0].page_content)} 字符")
    print(f"元数据: {docs[0].metadata}")
    print("✅ 文档加载测试通过")


def test_splitter():
    """测试分块模块"""
    print("\n" + "=" * 50)
    print("✂️  测试分块模块")
    print("=" * 50)

    from ingestion.loader import DocumentLoader
    from ingestion.splitter import TextSplitter

    loader = DocumentLoader()
    docs = loader.load_file("data/sample_docs/ai_basics.md")

    splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    stats = splitter.get_stats(docs, chunks)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("✅ 分块测试通过")


def test_vectorstore():
    """测试向量存储模块"""
    print("\n" + "=" * 50)
    print("💾 测试向量存储模块")
    print("=" * 50)

    from ingestion.loader import DocumentLoader
    from ingestion.splitter import TextSplitter
    from vectorstore.store import VectorStoreManager

    loader = DocumentLoader()
    docs = loader.load_file("data/sample_docs/ai_basics.md")
    splitter = TextSplitter()
    chunks = splitter.split_documents(docs)

    manager = VectorStoreManager(persist_dir="./test_chroma_db")
    manager.create_from_documents(chunks)

    stats = manager.get_stats()
    print(f"  向量数据库统计: {stats}")

    # 清理测试数据
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")

    print("✅ 向量存储测试通过")


def test_retrieval():
    """测试检索模块"""
    print("\n" + "=" * 50)
    print("🔍 测试检索模块")
    print("=" * 50)

    from ingestion.loader import DocumentLoader
    from ingestion.splitter import TextSplitter
    from vectorstore.store import VectorStoreManager
    from retrieval.retriever import SmartRetriever

    # 构建临时知识库
    loader = DocumentLoader()
    docs = loader.load_file("data/sample_docs/ai_basics.md")
    splitter = TextSplitter()
    chunks = splitter.split_documents(docs)
    manager = VectorStoreManager(persist_dir="./test_chroma_db")
    manager.create_from_documents(chunks)

    # 测试检索
    retriever = SmartRetriever(vectorstore=manager.get_vectorstore())

    query = "什么是RAG？"
    results = retriever.retrieve(query, strategy="mmr")
    print(f"\n  查询: {query}")
    print(f"  检索到 {len(results)} 个文档:")
    for i, doc in enumerate(results, 1):
        print(f"    [{i}] {doc.page_content[:80]}...")

    # 清理
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")

    print("✅ 检索测试通过")


def test_qa_chain():
    """测试 QA 链"""
    print("\n" + "=" * 50)
    print("🤖 测试 QA 链")
    print("=" * 50)

    from ingestion.loader import DocumentLoader
    from ingestion.splitter import TextSplitter
    from vectorstore.store import VectorStoreManager
    from retrieval.retriever import SmartRetriever
    from chain.qa_chain import QAChain

    # 构建临时知识库
    loader = DocumentLoader()
    docs = loader.load_file("data/sample_docs/ai_basics.md")
    splitter = TextSplitter()
    chunks = splitter.split_documents(docs)
    manager = VectorStoreManager(persist_dir="./test_chroma_db")
    manager.create_from_documents(chunks)

    # 测试问答
    retriever = SmartRetriever(vectorstore=manager.get_vectorstore())
    qa_chain = QAChain(retriever=retriever)

    question = "RAG系统的工作流程是什么？"
    answer, source_docs = qa_chain.answer(question)

    print(f"\n  问题: {question}")
    print(f"  答案: {answer[:200]}...")
    print(f"  引用了 {len(source_docs)} 个文档")

    # 清理
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")

    print("✅ QA 链测试通过")


if __name__ == "__main__":
    print("🧪 RAG 知识库系统 - 模块测试")
    print("=" * 50)

    tests = [
        ("文档加载", test_loader),
        ("文本分块", test_splitter),
        ("向量存储", test_vectorstore),
        ("智能检索", test_retrieval),
        ("QA 问答链", test_qa_chain),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name} 测试失败: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)
