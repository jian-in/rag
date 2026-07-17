"""
多模态视觉分析模块

为 RAG 系统添加图像理解能力。
面试亮点：
  - 理解多模态 LLM 的工作原理
  - 图文混合输入的 API 调用方式
  - 将视觉能力与 RAG 结合的创新思路

支持两种模式：
  1. 纯图像理解：用户上传图片，问关于图片的问题
  2. 图文混合 RAG：先理解图片内容，再结合知识库回答
"""

import base64
import os
from pathlib import Path
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import config


class VisionAnalyzer:
    """
    视觉分析器

    使用多模态 LLM（如 qwen-vl-max）来理解图像内容。
    可以和知识库结合：先用视觉模型描述图片，再用描述去检索知识库。
    """

    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Args:
            model_name: 多模态模型名称（如 qwen-vl-max, gpt-4o 等）
            api_key: API Key，默认使用 config 中的 OPENAI_API_KEY
            api_base: API Base URL，默认使用 config 中的 OPENAI_API_BASE
        """
        llm_kwargs = {
            "model": model_name,
            "temperature": 0,
            "openai_api_key": api_key or config.OPENAI_API_KEY,
        }
        api_base = api_base or config.OPENAI_API_BASE
        if api_base:
            llm_kwargs["openai_api_base"] = api_base

        self.llm = ChatOpenAI(**llm_kwargs)
        self.model_name = model_name
        print(f"✅ 视觉模型已加载: {model_name}")

    def analyze_image(
        self,
        image_path: str,
        question: str = "请详细描述这张图片的内容。",
    ) -> str:
        """
        分析图片并回答问题

        Args:
            image_path: 图片文件路径
            question: 关于图片的问题

        Returns:
            str: 视觉模型的回答
        """
        # 读取图片并转为 base64
        base64_image = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        # 构建多模态消息（兼容 OpenAI Vision API 格式）
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                    },
                },
            ]
        )

        response = self.llm.invoke([message])
        return response.content

    def image_to_text(self, image_path: str) -> str:
        """
        将图片转换为文字描述（用于后续 RAG 检索）

        这是将视觉能力与 RAG 结合的关键步骤：
          图片 → 文字描述 → 用描述去检索知识库 → 生成更丰富的答案

        Args:
            image_path: 图片文件路径

        Returns:
            str: 图片的结构化描述
        """
        prompt = (
            "请对这张图片进行详细描述，包括：\n"
            "1. 图片的主题和场景\n"
            "2. 图片中的关键对象和元素\n"
            "3. 如果图片包含文字，请提取所有文字内容\n"
            "4. 图片传达的信息或含义\n\n"
            "请用中文回答，尽量详细。"
        )
        return self.analyze_image(image_path, question=prompt)

    def image_rag_query(
        self,
        image_path: str,
        user_question: str,
    ) -> Tuple[str, str]:
        """
        图文结合的 RAG 查询

        流程：
          1. 先理解图片内容，生成描述
          2. 将图片描述 + 用户问题组合成增强的查询
          3. 返回（增强查询, 图片描述），增强查询用于后续知识库检索

        面试亮点：这是 RAG + 多模态的创新结合方式

        Args:
            image_path: 图片文件路径
            user_question: 用户关于图片的问题

        Returns:
            Tuple[str, str]: (增强后的检索查询, 图片描述)
        """
        # 先理解图片
        image_description = self.image_to_text(image_path)

        # 组合成增强查询
        enhanced_query = (
            f"根据以下图片内容：\n{image_description}\n\n"
            f"用户的问题是：{user_question}"
        )

        return enhanced_query, image_description

    def _encode_image(self, image_path: str) -> str:
        """将图片文件编码为 base64 字符串"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """根据文件扩展名获取 MIME 类型"""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return media_types.get(suffix, "image/png")
