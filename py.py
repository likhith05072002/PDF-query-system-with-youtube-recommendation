import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from googleapiclient.discovery import build 
from gtts import gTTS 
from io import BytesIO

# ✅ Load environment variables
load_dotenv()

# ✅ Correct API Key loading
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Creates a vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks were created. Ensure your PDFs contain extractable text.")
        return
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
        vector_store.save_local("faiss_index")
        st.success("Done.")
    except Exception as e:
        st.error(f"An error occurred during FAISS index creation: {str(e)}")

def get_conversational_chain():
    """Sets up the conversational chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If it is not related, say: "Sorry, answer not available in the context but here is what I know." 
    Also, if there are same topic names but different info, display "In this context there are same topics with different information" 
    and ask the user to specify the topic with 2-line info about each.

    Context:
    {context}?
    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def summarize_text(user_question, response):
    """Generates a YouTube search query based on the response."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
    summary_prompt = f"""Take the user's question and the AI's response, then generate a concise, relevant search query for YouTube.

    Question: {user_question}
    Response: {response}

    YouTube Search Query:"""
    search_query = model.predict(summary_prompt)
    return search_query.strip()

def search_youtube(query):
    """Searches YouTube for a relevant video."""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(q=query, part='snippet', maxResults=1, type='video')
        response = request.execute()

        if response['items']:
            video_id = response['items'][0]['id']['videoId']
            video_title = response['items'][0]['snippet']['title']
            return video_id, video_title
    except Exception as e:
        st.error(f"An error occurred with YouTube API: {str(e)}")
    return None, None

def synthesize_speech(text):
    """Converts text to speech."""
    tts = gTTS(text)
    output = BytesIO()
    tts.write_to_fp(output)
    output.seek(0)
    return output

def user_input(user_question, chat_history):
    """Processes user input and generates a response."""
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        try:
            new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer_text = response["output_text"]

            if "Sorry, answer not available" in answer_text:
                model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
                answer_text = model.predict(user_question)

            chat_history.append((user_question, answer_text))

            st.write(f"🤖 Reply: {answer_text}")

            # Add synthesized speech playback
            audio_data = synthesize_speech(answer_text)
            st.audio(audio_data, format='audio/mp3')

            # Add option to download the response
            st.download_button(
                label="Download Response",
                data=answer_text,
                file_name="response.txt",
                mime="text/plain",
            )

            # Generate and display YouTube search suggestions
            summary = summarize_text(user_question, answer_text)
            st.info(f"YouTube search suggestion: {summary}")
            video_id, video_title = search_youtube(summary)

            if video_id:
                st.subheader("Recommended YouTube Video:")
                st.video(f"https://www.youtube.com/watch?v={video_id}")
                st.caption(video_title)
            else:
                st.warning("No relevant video found on YouTube.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
    else:
        st.error("FAISS index file not found. Please upload your PDFs and process them first.")

def main():
    st.set_page_config("Chat with Multiple PDF and YouTube Integration")
    st.header("📚 PDF Query System with 🎥 YouTube Suggestions")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question related to your PDF content:")
    if user_question:
        user_input(user_question, st.session_state.chat_history)

    # Sidebar content
    with st.sidebar:
        st.title("📂 Menu:")
        pdf_docs = st.file_uploader("Upload your PDF(s) and click Submit", accept_multiple_files=True)
        if st.button("Submit"):
            if pdf_docs:
                with st.spinner("Processing your PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
            else:
                st.error("Please upload at least one PDF file.")

        # Chat history toggle
        if st.button("View Chat History"):
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.write(f"😃 **Q{i+1}:** {question}")
                st.write(f"🤖 **A{i+1}:** {answer}")

if __name__ == "__main__":
    main()
