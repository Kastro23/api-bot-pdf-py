from flask import Flask, request, jsonify
import os
import tempfile
import traceback
import filetype

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain  # Esto sigue siendo útil si usas otros métodos también
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from flask_cors import CORS


from dotenv import load_dotenv

# ✅ Cargar variables de entorno
load_dotenv(dotenv_path="./.env")


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:4200"}})
# Configura tu LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.5,
    convert_system_message_to_human=True
)

# Prompt
prompt = PromptTemplate.from_template(
    """
    Eres un asistente inteligente especializado en informes medioambientales, normativas legales y documentos técnicos. Tu función es analizar y responder preguntas basándote en el contenido de los documentos proporcionados por el usuario (generalmente PDFs de leyes, informes técnicos o normativas), siempre con claridad, precisión y profesionalismo.

Formato de respuesta:
- Usa un lenguaje natural, técnico pero accesible.
- Si el usuario solicita una respuesta **corta o resumen**, limita la respuesta a 2 o 3 frases clave, sin perder el sentido esencial.
- Si solicita una respuesta **larga o detallada**, desarrolla el contenido en párrafos organizados, claros y coherentes.
- No uses etiquetas tipo "Respuesta:", "Normativa:", "Resumen:", etc., a menos que el usuario lo indique explícitamente.
- Evita repetir innecesariamente partes de la pregunta en la respuesta.
- Si el documento no menciona algo directamente, acláralo con frases como: "No se menciona explícitamente, pero se puede inferir que...", o "No se encuentra una norma específica, pero se alude a...".
- Usa estilo de redacción elegante: frases fluidas, conectores apropiados, evitando listas a menos que el usuario las pida.
- Responde siempre como si formaras parte del equipo técnico de Nakamura Consultores.

Ejemplos de tono:
- Claro, directo, profesional.
- Sin exageraciones, sin adornos innecesarios.
- Si una respuesta no se puede generar, indica de forma amable que no hay información suficiente en el documento.

Importante:
- No uses etiquetas como Markdown (**negritas**, _cursivas_, `código`, etc.) a menos que el entorno de visualización lo permita (como en web estilizada con Tailwind).
- Ajusta automáticamente el tamaño del texto a la extensión pedida por el usuario.

Contexto adicional:
El usuario puede subir documentos legales (como leyes peruanas, normativas, resoluciones) y hacer preguntas técnicas sobre su contenido. Si hay referencias legales, debes citarlas con precisión. Si no las hay, señala su ausencia sin inventar.

Tu objetivo: Brindar respuestas útiles, elegantes y alineadas al estilo técnico de Nakamura Consultores.
Documentos base:
    {context}

    Pregunta:
    {input}
    """
)



# ✅ Ruta principal
@app.route('/api/pdf-chat', methods=['POST'])
def handle_pdf():
    try:
        # 1. Obtener archivo y pregunta
        uploaded_file = request.files['file']
        question = request.form['question']

        # 2. Guardar temporalmente el archivo
        extension = uploaded_file.filename.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
            file_path = tmp.name
            uploaded_file.save(file_path)

        # 3. Detectar tipo de archivo
        kind = filetype.guess(file_path)
        if not kind:
            return jsonify({"error": "No se pudo detectar el tipo de archivo."}), 400

        if kind.mime == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif kind.mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            return jsonify({"error": f"Tipo de archivo no soportado: {kind.mime}"}), 400

        # 4. Cargar y dividir el contenido
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # 5. Vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        retriever = vectorstore.as_retriever()

        # 6. Crear cadena de recuperación (retrieval chain)
        retrieval_chain = (
                RunnableMap({
                    "context": lambda x: retriever.invoke(x) if isinstance(x, str) else retriever.invoke(x["input"]),
                    "input": lambda x: x if isinstance(x, str) else x["input"]
                })
                | (lambda d: {
                    "context": "\n\n".join([doc.page_content for doc in d["context"]]),
                    "input": d["input"]
                })
                | prompt
                | llm
                | StrOutputParser()
            )



        # 7. Ejecutar la pregunta
        answer = response = retrieval_chain.invoke({"input": question})
        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
