"""
Gradio 界面样式与主题

从 main.py 抽离出来的自定义 CSS 和主题定义，让 main.py 专注于界面装配与业务逻辑。
"""

import gradio as gr


CUSTOM_CSS = """
/* ===== 全局暗色样式 ===== */
.gradio-container {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #0f1117 !important;
    min-height: 100vh;
    padding: 16px !important;
    color: #e2e8f0 !important;
}

/* ===== 顶部标题区域 ===== */
.header-section {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(79, 70, 229, 0.25);
    position: relative;
    overflow: hidden;
}
.header-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.header-section::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.header-section h1 {
    margin: 0 0 8px 0;
    font-size: 1.8em;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.header-section p {
    margin: 0;
    opacity: 0.85;
    font-size: 0.95em;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ===== 标签页 (胶囊风格 暗色) ===== */
.tab-nav {
    display: flex !important;
    gap: 4px !important;
    background: #1a1c2e !important;
    border-radius: 16px !important;
    padding: 6px !important;
    margin-bottom: 20px !important;
    border: 1px solid #2d2f45 !important;
}
.tab-nav button {
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #8b8fa3 !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
}
.tab-nav button:hover {
    background: #252740 !important;
    color: #a78bfa !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
}

/* ===== 聊天气泡 ===== */
.bot {
    background: #1e2035 !important;
    border: 1px solid #2d2f45 !important;
    border-radius: 16px 16px 16px 4px !important;
    color: #e2e8f0 !important;
    padding: 16px 20px !important;
}
.user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border-radius: 16px 16px 4px 16px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25) !important;
    padding: 16px 20px !important;
}

/* 聊天框容器 */
[class*="chatbot"] {
    border-radius: 20px !important;
    border: 1px solid #2d2f45 !important;
    background: #13141f !important;
}

/* ===== 输入框 ===== */
textarea, input[type="text"] {
    background: #1a1c2e !important;
    border: 2px solid #2d2f45 !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    transition: all 0.3s ease !important;
    padding: 12px 18px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
    outline: none !important;
    background: #1e2035 !important;
}

/* ===== 按钮 ===== */
.primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    color: white !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
    transition: all 0.3s ease !important;
}
.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(79, 70, 229, 0.4) !important;
}

/* ===== 面板和区块 ===== */
[class*="block"] {
    border-radius: 16px !important;
    border-color: #2d2f45 !important;
}

/* Markdown 区域 */
.markdown {
    border-radius: 16px !important;
    color: #cbd5e1 !important;
}

/* 代码块 */
code {
    background: #1a1c2e !important;
    color: #a78bfa !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}
pre {
    background: #1a1c2e !important;
    border: 1px solid #2d2f45 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
}

/* ===== Checkbox ===== */
input[type="checkbox"] {
    accent-color: #6366f1 !important;
}

/* ===== 文件上传区域 ===== */
[class*="upload"] {
    border: 2px dashed #3d3f5c !important;
    border-radius: 16px !important;
    background: #13141f !important;
    transition: all 0.3s ease !important;
}
[class*="upload"]:hover {
    border-color: #6366f1 !important;
    background: #1a1c2e !important;
}

/* ===== 表格 ===== */
table {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #2d2f45 !important;
    background: #1a1c2e !important;
}
th {
    background: #252740 !important;
    font-weight: 600 !important;
    color: #a78bfa !important;
}
td {
    border-top: 1px solid #2d2f45 !important;
    color: #cbd5e1 !important;
}

/* ===== 下拉框 / 选择器 ===== */
select, [role="listbox"] {
    background: #1a1c2e !important;
    border: 1px solid #2d2f45 !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #13141f;
}
::-webkit-scrollbar-thumb {
    background: #3d3f5c;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4f46e5;
}

/* ===== 动画 ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.header-section {
    animation: fadeIn 0.6s ease-out;
}
"""

CUSTOM_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Microsoft YaHei", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
)
