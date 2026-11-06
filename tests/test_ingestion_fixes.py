import tempfile
from pathlib import Path
import pytest
from langchain_core.documents import Document

from ingestion.loader import DocumentLoader
from ingestion.splitter import TextSplitter


def test_chinese_sentence_splitter_preserves_sentence_boundary():
    """中文分块测试：超长无换行段落应当在中文标点（句号等）处断开，而非硬截断汉字。"""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    
    # 构造由多个完整中文句子组成的长段落（无换行，无空格，总长约 220 字）
    s1 = "这是关于检索增强生成系统的第一条完整原理说明，用于验证文档分块的核心能力。"
    s2 = "第二条说明继续阐述语义检索与向量相似度计算的结合策略，保证上下文不丢失。"
    s3 = "第三条说明聚焦大语言模型在注入背景知识后的生成表现，提升专业领域的问答精度。"
    full_text = s1 + s2 + s3
    assert len(full_text) > 100
    
    docs = [Document(page_content=full_text, metadata={"source": "test.txt"})]
    chunks = splitter.split_documents(docs)
    
    # 期望至少切成了 2 个以上的 chunk
    assert len(chunks) >= 2
    # 关键断言：切分点应该落在中文标点处（如句号等），而不是生硬地切断句子
    # 检查第 1 个 chunk 的结尾，或者第 2 个 chunk 的开头是否尊重了句子边界
    valid_sentence_ends = ("。", "！", "？", "；", "\n")
    # 如果在标点处切开，第一块通常以标点结尾
    assert any(chunks[0].page_content.rstrip().endswith(punct) for punct in valid_sentence_ends), (
        f"分块未能在中文标点处切断，第1块结尾为: '{chunks[0].page_content[-10:]}'"
    )


def test_loader_utf8_bom_and_gbk_encoding():
    """文档加载测试：应当能够成功加载带 UTF-8 BOM 和 GBK 编码的文件。"""
    loader = DocumentLoader()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 1. UTF-8 BOM 文件
        bom_file = tmp_path / "test_bom.txt"
        bom_text = "这是包含UTF-8 BOM签名的文档内容。"
        bom_file.write_bytes(bom_text.encode("utf-8-sig"))
        
        docs_bom = loader.load_file(str(bom_file))
        assert len(docs_bom) == 1
        # 期望内容干净，不残留 BOM 字符 \ufeff
        assert not docs_bom[0].page_content.startswith("\ufeff")
        assert "这是包含UTF-8 BOM" in docs_bom[0].page_content
        
        # 2. GBK / GB18030 编码文件
        gbk_file = tmp_path / "test_gbk.txt"
        gbk_text = "这是GBK中文编码的文档，测试Windows历史格式容错能力。"
        gbk_file.write_bytes(gbk_text.encode("gb18030"))
        
        # 当前 loader.read_text(encoding="utf-8") 会直接爆 UnicodeDecodeError
        docs_gbk = loader.load_file(str(gbk_file))
        assert len(docs_gbk) == 1
        assert "这是GBK中文编码" in docs_gbk[0].page_content
