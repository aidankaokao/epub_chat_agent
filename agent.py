import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Any
from operator import add
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils_epub import parse_epub

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# ===
# basic function
# ===
# def _format_docs(docs) -> str:
#     # 將搜尋到的 Documents 合併成單一 context 字串
#     return "\n\n".join(d.page_content for d in docs)

class EpubChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages()]
    epub_path: str
    text_chunks: Annotated[List[str], add]
    summary: str
    vectorstore: Any

class EpubChatAgent:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL_NAME)
        self.embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

    async def create_agent(self):
        # ===
        # node function
        # ===

        def parse_epub_node(state: EpubChatState):
            """解析 EPUB 並切割成 chunks"""
            print("正在解析 EPUB...")
            texts = parse_epub(state["epub_path"])
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
            chunks = splitter.split_text("\n".join(texts))
            return {"text_chunks": chunks}
        
        def summarize_or_index_node(state: EpubChatState):
            """摘要（先取前 10 段）與建立向量索引"""
            print("產生摘要 & 建立向量索引 ...")

            prompt = ChatPromptTemplate.from_template("請用繁體中文為以下內容生成摘要：\n\n{text}")
            chain = prompt | self.llm | StrOutputParser()

            summaries = []
            for chunk in state["text_chunks"][:10]:
                s = chain.invoke({"text": chunk}).strip()
                if s:
                    summaries.append(s)
            summary = "\n".join(summaries)

            store = FAISS.from_texts(state["text_chunks"], embedding=self.embedding)

            print("摘要與索引完成")
            return {"summary": summary, "vectorstore": store}
        
        def qa_node(state: EpubChatState):
            """基於檢索的問答（修正版，防止 list | list 錯誤）"""
            print("💬 問答中 ...")

            retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 5})

            prompt = ChatPromptTemplate.from_template(
                "以下是書籍內容節選，請根據它們回答問題：\n\n{context}\n\n問題：{input}\n\n請用繁體中文、條理清楚地回答。"
            )

            # ✅ 這裡明確定義 retriever_chain，確保 _format_docs 接收的是 list 並回傳 string
            def retrieve_context(question: str) -> str:
                docs = retriever.invoke(question)
                return "\n\n".join(d.page_content for d in docs)

            # ✅ RAG chain 組裝
            rag_chain = (
                {
                    "context": retrieve_context,
                    "input": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

            last_msg = state["messages"][-1]
            question = last_msg.content if isinstance(last_msg, HumanMessage) else ""
            answer = rag_chain.invoke(question).strip()

            print("✅ 問答完成")
            return {"messages": state["messages"] + [AIMessage(content=answer)]}

        # ===
        # Build graph
        # ===
        graph = StateGraph(EpubChatState)
        # add node
        graph.add_node("parse_epub", parse_epub_node)
        graph.add_node("summarize_or_index", summarize_or_index_node)
        graph.add_node("qa", qa_node)
        # add edge
        graph.add_edge(START, "parse_epub")
        graph.add_edge("parse_epub", "summarize_or_index")
        graph.add_edge("summarize_or_index", "qa")
        graph.add_edge("qa", END)
        # compile
        agent = graph.compile()

        return agent


async def main():
    aiagent_client = EpubChatAgent()
    agent = await aiagent_client.create_agent()

    config = {"configurable": {"thread_id": "epub-session-001"}, "recursion_limit": 1000}
    epub_path = "epub/test.epub"
    question = "請幫我摘要這本書的主要內容與核心論點"
    init_state = {
        "epub_path": epub_path,
        "messages": [HumanMessage(content=question)],
    }
    result = await agent.ainvoke(input=init_state, config=config)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())