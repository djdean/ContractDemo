#Go through how to add context to the prompt. Specifically, read a PDF in, and use GPT to summarize the text.

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import tiktoken
import streamlit as st
from Utilities import Utils
import numpy as np
def main():
    
    config_data = Utils.read_json_data(r"C:\Users\dade\Desktop\ContractDemo\config\config.json")
    model = config_data["model"]
    openai_api_version = config_data["openai_api_version"]
    endpoint = config_data["endpoint"]
    openai_key = config_data["openai_key"]
    document_intelligence_endpoint = config_data["document_intelligence_endpoint"]
    document_intelligence_key = config_data["document_intelligence_key"]
    model_family = config_data["model_family"]
    embedding_name = config_data["embedding_name"]
    k = config_data["top_k"]
    chunk_size = config_data["chunk_size"]
    st.set_page_config(layout="wide")
    init_sidebar(model,openai_api_version,endpoint,openai_key, document_intelligence_endpoint,document_intelligence_key)
    uploaded_files = st.file_uploader("Select a file(s)", accept_multiple_files=True)
    if len(uploaded_files) > 0 and not "content" in st.session_state:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    if not "content" in st.session_state:
                        st.session_state["content"] = []
                        st.session_state["embeddings"] = []
                        st.session_state["embedding_content"] = []
                    document = uploaded_file.getvalue()
                    content = parse_pdf(document, st.session_state["DI_client"])
                    content["title"] = get_filename_pretty(uploaded_file.name)
                    content["type"] = uploaded_file.type
                    content["size"] = uploaded_file.size
                    st.session_state["content"].append(content)
                    document_status = get_embeddings(st.session_state["AOAI_client"],content,embedding_name,model_family,chunk_size)
                    if not document_status:
                        st.error("Error processing the document.")
    if "content" in st.session_state:
        if not "messages" in st.session_state:
                st.session_state["messages"] = []
        question = st.chat_input("Enter a question to ask "+model+ " about the uploaded documents:")
        messages = st.session_state["messages"]
        history = ""
        for message in messages:
            history += message['content']
            with st.chat_message(message['role']):
                if message['role'] == "Assistant":
                    st.markdown(message['content'])
                else:
                    st.markdown(message['content'])
        if question:
            top_k = get_top_k(question,embedding_name,k)
            context = ""
            for top_result in top_k:
                context+=top_result["text"]+"\n"
            handle_question(question,model,context,history)
def get_embeddings(client,content,embedding_name,model_family,chunk_size):
    text = content["paragraphs"] + content["tables"]
    chunks = Utils.get_semantic_chunks(text, model_family, chunk_size)
    for chunk in chunks:
        embedding = client.embeddings.create(input = [chunk], model=embedding_name).data[0].embedding
        st.session_state["embeddings"].append(embedding)
        content = {
            "text":chunk,
            "type":content["type"],
            "size":content["size"],
            "title":content["title"]
        }
        st.session_state["embedding_content"].append(content)
    return True
def get_top_k(input_text, embedding_name, k):
    text_embedding = st.session_state["AOAI_client"].embeddings.create(input = [input_text], model=embedding_name).data[0].embedding
    cosine_similarities = []
    similarity_map = {}
    for i in range(0,len(st.session_state["embeddings"])):
        current_similarity = cosine_similarity(text_embedding,st.session_state["embeddings"][i])
        cosine_similarities.append(current_similarity)
        similarity_map[current_similarity] = st.session_state["embedding_content"][i]
    top_k = []
    cosine_similarities.sort()
    for i in range(0,k):
        if i < len(cosine_similarities):
            top_k.append(similarity_map[cosine_similarities[i]])
        #Not enough chunks to get top k
        else:
            break
    return top_k

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def handle_question(question,model,paragraphs,history):
    answer = answer_question(question,st.session_state["AOAI_client"],model, paragraphs, history)
    user_message = {
        "role":"user",
        "content":question
    }
    st.session_state["messages"].append(user_message)
    response_message = {
        "role":"Assistant",
        "content":answer,
    }
    with st.chat_message(response_message["role"]):
        st.markdown(response_message['content'])
    st.session_state["messages"].append(response_message) 
def init_sidebar(model,openai_api_version,endpoint,openai_key, document_intelligence_endpoint,document_intelligence_key):
    with st.sidebar:
        st.header(':green[Azure OpenAI Configuration]')
        st.subheader(':orange[Model:] ' + model)
        st.subheader(':orange[API Version:] ' + openai_api_version)
        if not "AOAI_client" in st.session_state:
            st.markdown("Connecting to Azure...")
            client = AzureOpenAI(
                azure_endpoint = endpoint, 
                api_key=openai_key,  
                api_version=openai_api_version
            )
            document_intelligence_client = DocumentIntelligenceClient(
                endpoint=document_intelligence_endpoint, credential=AzureKeyCredential(document_intelligence_key)
            )
            st.session_state["DI_client"] = document_intelligence_client
            st.session_state["AOAI_client"] = client
            st.markdown("Done")
            
        else:
             st.markdown("Connected.")
        

def get_filename_pretty(path):
     path_split = path.split("\\")
     filename_only = path_split[len(path_split)-1]
     return filename_only
def get_num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
def parse_pdf(doc,document_intelligence_client):
        poller_layout = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=bytes(doc)), locale="en-US"
        )
        layout: AnalyzeResult = poller_layout.result()
        paragraph_content = ""
        table_content = ""
        for p in layout.paragraphs:
            paragraph_content += f"{p.content}\n"
        for t in layout.tables:
            previous_cell_row=0
            rowcontent='| '
            tablecontent = ''
            for c in t.cells:
                if c.row_index == previous_cell_row:
                    rowcontent +=  c.content + " | "
                else:
                    tablecontent += rowcontent + "\n"
                    rowcontent='|'
                    rowcontent += c.content + " | "
                    previous_cell_row += 1
            table_content += f"{tablecontent}\n"
        return_content = {
            "paragraphs": paragraph_content,
            "tables": table_content
        }
        return return_content
def answer_question(question,client, model,content, history):
        response = client.chat.completions.create(
            model=model, # model = "deployment_name".
            messages=[
                {"role": "system", "content": "You are an AI assistant extremely proficient in answering questions coming from different users based on input context. When possible, insert some humor into your responses."},
                {"role": "user", "content": "Based on the following context:\n\n"+content+"\n\n Along with the user's chat history:\n\n"+history+"\n\nAnswer the following question:\n\n"+question+""\
                },
            ]
        )
        return response.choices[0].message.content
           
if __name__ == "__main__":
    main()

