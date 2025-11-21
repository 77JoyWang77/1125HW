import os
from pathlib import Path
from dotenv import load_dotenv

# ==================== 導入 LangChain 模組 ====================
from langchain_community.document_loaders import PyPDFLoader  # PDF 加載器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割
from langchain_chroma import Chroma  # 向量資料庫
from langchain_huggingface import HuggingFaceEmbeddings  # 本地嵌入模型
from langchain_groq import ChatGroq  # Groq LLM
from langchain_core.prompts import ChatPromptTemplate  # Prompt 模板
from langchain_core.runnables import RunnablePassthrough  # 可運行物件
from langchain_core.output_parsers import StrOutputParser  # 輸出解析

load_dotenv()  # 加載 .env 文件中的環境變數

# ==================== RAG 系統主類 ====================
class PDFRAGSystem:
    """
    RAG (Retrieval-Augmented Generation) 系統
    流程：PDF → 分割 → 向量化 → 存儲 → 檢索 → LLM 回答
    """
    
    def __init__(self, groq_api_key=None, chroma_path="./chroma_db_groq"):
        """
        初始化 RAG 系統
        
        Args:
            groq_api_key: Groq API 金鑰（來自 https://console.groq.com）
            chroma_path: 向量資料庫本地存儲路徑
                        - 首次會建立，之後可重複使用
                        - 避免每次都重新向量化
        """
        # 1. 設定 Groq API 金鑰
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("請設定 GROQ_API_KEY 環境變數或傳入參數")
        
        # 2. 設定向量資料庫路徑
        self.chroma_path = chroma_path
        
        # 3. 初始化嵌入模型（用於把文本轉成向量）
        print("🚀 初始化 HuggingFace Embeddings（本地模型）...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # 22MB，速度快，足夠好用
        )
        print("✅ Embeddings 已加載")
        
        # 4. 初始化（稍後在加載 PDF 後會設定）
        self.vectorstore = None  # 向量資料庫
        self.retriever = None    # 檢索器
        self.chain = None        # RAG 鏈
        self._setup_chain()      # 建立空的鏈（會在加載向量庫後更新）
    
    def _setup_chain(self):
        """
        建立 RAG 鏈路（核心邏輯）
        流程：用戶問題 → 檢索相關文本 → 組合 Prompt → LLM 回答
        """
        # 1. 初始化 LLM（語言模型）
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # 免費快速模型
            api_key=self.api_key,
            temperature=0.7,                # 0-1，越高越有創意
            max_tokens=1000                 # 最多生成 1000 個 token
        )
        
        # 2. 定義 Prompt 模板
        template = """你是一個有幫助的助手，根據提供的文件內容回答問題。

        文件內容：
        {context}

        問題：{question}

        請基於上述文件內容回答問題。如果文件中沒有相關信息，請說明。"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 3. 格式化文檔函數（把檢索到的多個文檔合併）
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 4. 組建 RAG 鏈
        # 使用 LCEL (LangChain Expression Language) 語法
        self.chain = (
            {
                # 檢索相關文檔（如果有檢索器的話）
                "context": self.retriever | format_docs if self.retriever else RunnablePassthrough(),
                # 直接傳遞用戶問題
                "question": RunnablePassthrough()
            }
            | prompt      # 把 context 和 question 插入 prompt 模板
            | llm         # 發送給 LLM
            | StrOutputParser()  # 把輸出轉成字符串
        )
    
    # ==================== PDF 加載方法 ====================
    
    def load_pdf(self, pdf_path):
        """
        加載單個 PDF 文件
        
        Args:
            pdf_path: PDF 文件路徑
        
        Returns:
            Document 物件列表
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
        
        print(f"📖 正在加載 PDF：{pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # 每頁返回一個 Document 物件
        
        print(f"✅ 已加載 {len(pages)} 頁內容")
        return pages
    
    def load_multiple_pdfs(self, pdf_dir):
        """
        從目錄批量加載 PDF 文件
        
        Args:
            pdf_dir: 包含 PDF 的目錄路徑
        
        Returns:
            所有文檔的列表
        """
        pdf_path = Path(pdf_dir)
        if not pdf_path.is_dir():
            raise NotADirectoryError(f"目錄不存在：{pdf_dir}")
        
        all_pages = []
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ 在 {pdf_dir} 中未找到 PDF 文件")
            return all_pages
        
        for pdf_file in pdf_files:
            print(f"📖 正在加載：{pdf_file.name}")
            pages = self.load_pdf(str(pdf_file))
            all_pages.extend(pages)
        
        print(f"✅ 共加載了 {len(all_pages)} 頁內容")
        return all_pages
    
    # ==================== 向量化和存儲方法 ====================
    
    def create_vector_store(self, documents):
        """
        把文檔轉成向量並存儲
        
        步驟：
        1. 分割文檔（1000 字符一塊，200 字符重疊）
        2. 向量化每一塊
        3. 存儲到 Chroma 資料庫
        4. 建立檢索器
        
        Args:
            documents: Document 物件列表
        """
        # 第 1 步：分割文本
        print("\n🔄 正在分割文本...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # 每塊 1000 字符
            chunk_overlap=200,      # 相鄰塊重疊 200 字符（保留上下文）
            separators=["\n\n", "\n", " ", ""]  # 優先按段落、句子分割
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✅ 已分割成 {len(chunks)} 個文本塊")
        
        # 第 2-3 步：向量化並存儲
        print("\n🔄 正在建立向量資料庫...")
        print("   (首次加載嵌入模型可能需要時間)")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.chroma_path  # 持久化存儲
        )
        print(f"✅ 向量資料庫已建立，存儲在 {self.chroma_path}")
        
        # 第 4 步：建立檢索器（用於查找相似文本）
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        # k=3 表示每次查詢返回最相似的 3 個文本塊
        
        # 重新建立 RAG 鏈（現在有了檢索器）
        self._setup_chain()
    
    def load_existing_vectorstore(self):
        """
        加載已存在的向量資料庫（避免重複向量化）
        
        Returns:
            是否加載成功
        """
        if not Path(self.chroma_path).exists():
            print(f"⚠️ 向量資料庫不存在於 {self.chroma_path}")
            return False
        
        print(f"📂 正在加載向量資料庫...")
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self._setup_chain()
        print("✅ 向量資料庫已加載")
        return True
    
    # ==================== 查詢方法 ====================
    
    def query(self, question):
        """
        提出問題並獲取答案
        
        流程：
        1. 向量化問題
        2. 在向量資料庫中搜索相似文本
        3. 把文本和問題發給 LLM
        4. 返回 LLM 的回答
        
        Args:
            question: 用戶問題
        
        Returns:
            LLM 的答案
        """
        if not self.chain:
            raise ValueError("RAG 系統未初始化，請先加載 PDF")
        
        print(f"\n❓ 問題：{question}")
        print("🔄 正在思考中...")
        answer = self.chain.invoke(question)
        print(f"💬 答案：{answer}")
        return answer
    
    def interactive_chat(self):
        """
        進入互動對話模式
        用戶可以持續提問，輸入 'exit' 退出
        """
        print("\n🤖 進入互動式對話模式（輸入 'exit' 退出）\n")
        
        while True:
            try:
                question = input("👤 你的問題：").strip()
                if question.lower() == 'exit':
                    print("👋 再見！")
                    break
                if not question:
                    continue
                self.query(question)
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break


# ==================== 主程式入口 ====================

def main():
    """
    主程式流程：
    1. 初始化 RAG 系統
    2. 加載 PDF（優先級：sample.pdf > ./pdfs > 現有資料庫）
    3. 執行測試查詢
    4. 進入互動對話
    """
    
    # 第 0 步：初始化
    print("=" * 50)
    print("🚀 PDF RAG 系統啟動")
    print("=" * 50)
    
    try:
        rag = PDFRAGSystem()
    except ValueError as e:
        print(f"❌ 錯誤：{e}")
        return
    
    # 第 1 步：加載 PDF
    print("\n" + "=" * 50)
    print("第一步：加載 PDF 文件")
    print("=" * 50)
    
    # 嘗試方式 A：加載 sample.pdf
    pdf_file = "sample.pdf"
    if Path(pdf_file).exists():
        documents = rag.load_pdf(pdf_file)
        rag.create_vector_store(documents)
    else:
        # 嘗試方式 B：加載 ./pdfs 目錄中的所有 PDF
        pdf_dir = "./pdfs"
        if Path(pdf_dir).exists():
            documents = rag.load_multiple_pdfs(pdf_dir)
            if documents:
                rag.create_vector_store(documents)
        else:
            # 嘗試方式 C：使用已存的向量資料庫
            if not rag.load_existing_vectorstore():
                print("\n❌ 請提供 PDF 文件")
                print("   方式 1：在當前目錄放 sample.pdf")
                print("   方式 2：在 ./pdfs 目錄放 PDF 文件")
                print("   方式 3：使用已建立的 chroma_db_groq")
                return
    
    # 第 2 步：執行測試查詢
    print("\n" + "=" * 50)
    print("第二步：進行問答")
    print("=" * 50)
    
    test_questions = [
        "文件的主要內容是什麼？",
        "請總結文件中的關鍵信息",
    ]
    
    for q in test_questions:
        try:
            rag.query(q)
        except Exception as e:
            print(f"❌ 查詢時出錯：{e}")
    
    # 第 3 步：進入互動對話
    rag.interactive_chat()


# ==================== 程式入口 ====================

if __name__ == "__main__":
    main()
