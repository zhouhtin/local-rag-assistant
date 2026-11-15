# ===================================================================
# FREE LANGCHAIN RESEARCH ASSISTANT - NO API KEYS REQUIRED!
# ===================================================================

# Standard library imports
import os                    # For file operations
import tempfile             # For temporary files
from typing import List, Dict, Any  # For type hints

# Third-party imports
import streamlit as st      # Web framework
from dotenv import load_dotenv  # Environment variables (optional)
import torch               # For checking if GPU is available

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Free AI model imports
from sentence_transformers import SentenceTransformer  # Free embeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # Free language models
from langchain.embeddings import HuggingFaceEmbeddings  # Free embeddings for LangChain
from langchain.llms import HuggingFacePipeline  # Free LLM integration

# ===================================================================
# CONFIGURATION
# ===================================================================

# Load environment variables (optional since we're using free models)
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="Free AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# FREE AI RESEARCH ASSISTANT CLASS
# ===================================================================

class FreeResearchAssistant:
    """
    A completely free research assistant using Hugging Face models.
    No API keys or paid services required!
    """
    
    def __init__(self):
        """Initialize the free research assistant."""
        
        # Display initialization message
        st.info("🚀 Initializing free AI models... This may take a moment on first run.")
        
        # Pick the best available device: NVIDIA GPU, Apple GPU, or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            st.success("🚀 NVIDIA GPU detected! This will be faster.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            st.success("🍎 Apple GPU (MPS) detected! This will be faster.")
        else:
            self.device = "cpu"
            st.info("💻 Using CPU. This might be slower but will work fine!")
        
        # Initialize components
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.conversation_chain = None
        
        # Setup memory
        self.memory = ConversationBufferWindowMemory(
            k=3,  # Remember last 3 exchanges (reduced for free tier)
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize AI models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize free AI models from Hugging Face."""
        
        try:
            # Step 1: Initialize free embeddings model
            st.info("📊 Loading free embedding model...")
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={'device': self.device}
            )
            st.success("✅ Embedding model loaded!")
            
            # Step 2: Initialize free language model
            st.info("🧠 Loading free language model... (This may take a few minutes on first run)")

            model_name = os.getenv("FREE_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )

            # Create text generation pipeline
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                return_full_text=False,
                device=self.device
            )

            # Wrap in LangChain
            self.llm = HuggingFacePipeline(pipeline=text_generator)

            st.success("✅ Language model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.info("💡 Tip: Try restarting the app or check your internet connection")
    
    def process_documents(self, uploaded_files: List) -> bool:
        """Process documents using free models."""
        
        if not self.embeddings:
            st.error("❌ Models not initialized properly!")
            return False
        
        try:
            # Step 1: Load all documents
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                # Load based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(temp_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    loader = TextLoader(temp_path, encoding='utf-8')
                else:
                    st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
                    os.unlink(temp_path)
                    continue
                
                # Load documents
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata['source_file'] = uploaded_file.name
                
                all_documents.extend(documents)
                os.unlink(temp_path)  # Clean up
            
            if not all_documents:
                st.error("❌ No documents to process!")
                return False
            
            # Step 2: Split documents
            st.info("✂️ Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,    # Smaller chunks for free models
                chunk_overlap=100,  # Less overlap to reduce processing time
                length_function=len
            )
            
            chunks = text_splitter.split_documents(all_documents)
            st.success(f"✅ Created {len(chunks)} chunks from {len(all_documents)} pages")
            
            # Step 3: Create vector store (this might take a moment)
            st.info("🔢 Creating searchable database... (This may take a few minutes)")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process in smaller batches for better performance
            batch_size = 10
            all_vectors = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(chunks) + batch_size - 1) // batch_size
                
                status_text.text(f"Processing batch {batch_num}/{total_batches}")
                progress_bar.progress(i / len(chunks))
                
                # Create vector store for this batch
                if i == 0:
                    # First batch - create new vector store
                    self.vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    # Subsequent batches - add to existing vector store
                    batch_vectors = FAISS.from_documents(batch, self.embeddings)
                    self.vectorstore.merge_from(batch_vectors)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Vector database created!")
            
            # Step 4: Create conversation chain
            st.info("🔗 Setting up conversation chain...")
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),  # Fewer results for speed
                memory=self.memory,
                return_source_documents=True,
                verbose=False  # Reduce output for cleaner interface
            )
            
            st.success("🎉 Documents processed successfully with free models!")
            return True
            
        except Exception as e:
            st.error(f"💥 Error processing documents: {str(e)}")
            st.info("💡 Try with smaller documents or restart the app")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question using free models."""
        
        if not self.conversation_chain:
            return {
                "answer": "⚠️ Please upload and process documents first!",
                "source_documents": []
            }
        
        try:
            # Show processing message
            with st.spinner("🤔 Thinking... (Free models take a bit longer)"):
                response = self.conversation_chain({"question": question})
            
            return response
            
        except Exception as e:
            return {
                "answer": f"❌ Error: {str(e)}. Try a simpler question or restart the app.",
                "source_documents": []
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

# ===================================================================
# STREAMLIT INTERFACE
# ===================================================================

def main():
    """Main application interface."""
    
    # Header
    st.title("🎓 Free AI Research Assistant")
    st.markdown("""
    **Completely free AI-powered document assistant for students!**
    
    ✅ **No API keys required**  
    ✅ **No paid services**  
    ✅ **Runs on your computer**  
    ✅ **Perfect for learning**
    
    ---
    """)
    
    # Important note for students
    st.info("""
    📚 **Student Note**: This uses free AI models from Hugging Face. 
    The first run will download models (~500MB) but then works offline!
    """)
    
    # Initialize assistant
    if 'free_assistant' not in st.session_state:
        with st.spinner("🔄 Starting free AI assistant..."):
            st.session_state.free_assistant = FreeResearchAssistant()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files (PDF or TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload your study materials, research papers, etc."
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                success = st.session_state.free_assistant.process_documents(uploaded_files)
                if success:
                    st.session_state.documents_processed = True
                    st.balloons()
        
        st.divider()
        
        st.header("🧠 Memory")
        if st.button("🧹 Clear Memory"):
            st.session_state.free_assistant.clear_memory()
            st.session_state.chat_history = []
            st.success("Memory cleared!")
        
        st.divider()
        
        st.header("💡 Tips for Free Models")
        st.markdown("""
        - **First run**: Takes longer to download models
        - **Responses**: May be slower than paid APIs
        - **Questions**: Keep them clear and specific
        - **Documents**: Smaller files work better
        - **Patience**: Free models need more time!
        """)
        
        st.header("🎯 Perfect For")
        st.markdown("""
        - 📚 Study materials
        - 📄 Research papers  
        - 📝 Class notes
        - 📖 Textbook chapters
        - 🎓 Academic projects
        """)
    
    # Main interface
    if hasattr(st.session_state, 'documents_processed'):
        st.header("💬 Ask Questions")
        
        question = st.text_input(
            "What would you like to know?",
            placeholder="Ask about your documents...",
            help="Ask questions about the uploaded documents"
        )
        
        if st.button("🔍 Ask", type="primary") and question:
            result = st.session_state.free_assistant.ask_question(question)
            
            # Display answer
            st.subheader("💡 Answer:")
            st.write(result["answer"])
            
            # Display sources
            if result.get("source_documents"):
                st.subheader("📚 Sources:")
                for i, doc in enumerate(result["source_documents"][:2]):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source_file', 'Unknown')}"):
                        st.write(doc.page_content[:400] + "...")
            
            # Add to history
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"]
            })
    
    else:
        st.header("🚀 Get Started")
        st.markdown("""
        ### How to use this free assistant:
        
        1. **Upload** your documents (PDF or text files)
        2. **Click** "Process Documents" 
        3. **Wait** for processing (first time takes longer)
        4. **Ask** questions about your documents
        5. **Get** AI-powered answers for free!
        
        ### What makes this special:
        - 🆓 **Completely free** - no subscriptions or API costs
        - 🔒 **Privacy-focused** - your documents stay on your computer
        - 🎓 **Student-friendly** - designed for academic use
        - 📚 **Educational** - learn AI concepts hands-on
        """)
        
        # Sample questions for inspiration
        st.subheader("💡 Example Questions You Can Ask:")
        st.markdown("""
        - "What are the main topics covered in these documents?"
        - "Summarize the key points from chapter 3"
        - "What does the author say about [specific topic]?"
        - "Compare the different theories mentioned"
        - "What are the conclusions of this research?"
        """)
    
    # Chat history
    if st.session_state.chat_history:
        st.header("📜 Recent Questions")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
                st.write(f"**Q:** {chat['question']}")
                st.write(f"**A:** {chat['answer']}")

if __name__ == "__main__":
    main()