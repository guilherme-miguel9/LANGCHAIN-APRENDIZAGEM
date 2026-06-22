from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


modelo = ChatOpenAI(
    model='google/gemma-4-e4b',
    temperature=0.5,
    api_key='llm',
    base_url='http://127.0.0.1:1234/v1'
)

#Buscar pedacos do texto que são mais relevantes a EMBEDDINGS, transforma nossa frases/palavra em vetores para serem mais facilmente encontrado pela nossa IA generativa
embeddings = OpenAIEmbeddings()

arquivos = []

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
    
)


documento = TextLoader(
    '',
    encoding='utf-8'
).load()

pedacos = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap=100
).split_documents(documento)

dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={'k' : 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ('system', 'Responda usando exclusivam,ente o conteúdo fornecido'),
        ('human', '{query}\n\n Contexto: \n {contexto}\n\n Resposta:')
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = '\n\n'.join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke(
        {
            'query': pergunta, 'contexto' : contexto
        }
    )
    
print(responder('Como devo proceder caso tenha um item roubado? '))