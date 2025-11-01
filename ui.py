import streamlit as st
import asyncio
import uuid
import tempfile
from langchain_core.messages import HumanMessage
from agent import EpubChatAgent

async def main():
    # ===
    # Call agent
    # ===
    aiagent_client = EpubChatAgent()
    agent = await aiagent_client.create_agent()

    # ===
    # UI
    # ===
    st.set_page_config(page_title="📘 EPUB 摘要與問答 Agent", layout="centered")

    st.title("📘 EPUB 摘要與問答 Agent")

    st.markdown(
        """
        上傳 EPUB 電子書後，輸入問題即可讓 LLM 自動解析內容並回答。  
        _例如輸入：「請幫我摘要這本書的主要內容與核心論點」_
        """
    )

    uploaded_file = st.file_uploader("上傳 EPUB 檔案", type=["epub"])

    question = st.text_input(
        "輸入你的問題",
        placeholder="請幫我摘要這本書的主要內容與核心論點",
    )

    if st.button("開始分析", type="primary"):
        if uploaded_file is None:
            st.warning("請先上傳 EPUB 檔案！")
        elif not question.strip():
            st.warning("請輸入問題！")
        else:
            with st.spinner("正在解析並生成答案，請稍候..."):
                # 暫存上傳的 epub 檔案
                with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    epub_path = tmp_file.name

                # 建立 config
                config = {
                    "configurable": {
                        "thread_id": f"epub_session_{uuid.uuid4()}",
                    },
                    "recursion_limit": 1000,
                }

                # 初始化狀態
                init_state = {
                    "epub_path": epub_path,
                    "messages": [HumanMessage(content=question)],
                }

                # 執行 Graph
                try:
                    result = await agent.ainvoke(init_state, config=config)

                    summary = result.get("summary", "")
                    messages = result.get("messages", [])
                    final_answer = messages[-1].content if messages else ""

                    # 顯示結果
                    st.subheader("📘 書籍摘要")
                    st.write(summary or "（無摘要資料）")

                    st.subheader("💬 問答結果")
                    st.write(final_answer or "（無回答）")

                except Exception as e:
                    st.error(f"執行時發生錯誤：{e}")


if __name__ == "__main__":
    asyncio.run(main())