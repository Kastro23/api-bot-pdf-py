from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS

import os
import tempfile
import traceback

from dotenv import load_dotenv

# ✅ Cargar variables de entorno
load_dotenv(dotenv_path="./.env")

# ✅ Inicializar Flask
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}})

# ✅ Configurar clave de Gemini
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ✅ Definir modelo de lenguaje (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
    temperature=0.5,
    convert_system_message_to_human=True)

# ✅ Prompt
prompt = ChatPromptTemplate.from_template("""
Responde la siguiente pregunta de manera precisa usando el contexto extraído del documento:

Contexto:
{context}

Pregunta:
{input}
""")

# ✅ Ruta para subir PDF y hacer preguntas
@app.route('/api/pdf-chat', methods=['POST'])
def handle_pdf():
    try:
        # 1. Obtener archivo PDF
        pdf_file = request.files['file']
        question = request.form['question']

        # 2. Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            pdf_file.save(pdf_path)

        # 3. Cargar y dividir el contenido
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # 4. Embeddings + FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # 5. Crear cadena de recuperación con LLM
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        # 6. Ejecutar respuesta
        response = retrieval_chain.invoke({"input": question})
        return jsonify({"answer": response["answer"]})

    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
        app.debug = True
        app.run(host="0.0.0.0", port=5000)