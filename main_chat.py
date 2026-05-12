from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

modelo = ChatOpenAI(
    model = 'google/gemma-4-e4b',
    temperature= 0.5,
    api_key='llm',
    base_url='http://127.0.0.1:1234/v1'
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", 'voce e um guia de viagem especilizado em destinos brasileiros. Apresente-se como Sr. Passeios'),
        ('placeholder', '{historico}'),
        ('human', '{query}')
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = 'aula_langchain_alura'

def historico_por_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    'Quero visitar um lugar no brasil, famoso por praias e culturas.',
    'Qual a melhor epoca do ano para ir?'
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            'query' : uma_pergunta,
        },
        config={"configurable": {"session_id": sessao}}
    )
    print('Usuario: ', uma_pergunta)
    print('IA: ', resposta, '\n')