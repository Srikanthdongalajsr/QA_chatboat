from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS 
from langchain.llms import GooglePalm 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

api_key = "AIzaSyCReKsDaLyxgqYwUI_35bkezF3kSHlWKxU"
file_path = "codebasics_faqs.csv"

llm = GooglePalm(google_api_key=api_key, temperature=0.5)
embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_path = "faiss_index"

def load_data():
    loader = CSVLoader(file_path=file_path, encoding="latin1")
    data = loader.load()
    return data

def vectordb():
    data = load_data()
    vector_db = FAISS.from_documents(documents=data, embedding=embedding)
    vector_db.save_local(vectordb_path)
    return vector_db
    
def chatbot():
    vector_db = FAISS.load_local(vectordb_path, embedding)
    retriever = vector_db.as_retriever(score_threshold = 0.7)
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""
    
    prompt = PromptTemplate(
        template = prompt_template,
        input_variables= ["context", "question"]
    )
    
    # kwargs = {"prompt": prompt}
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": prompt})
    
    return chain

if __name__ == "__main__":
    vectordb()
    chain = chatbot()
    print(chain("Whats the duration of the course?"))
    