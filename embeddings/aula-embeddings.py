

# Agora é a sua chance de aplicar os conceitos desta aula. Para isso:

#     Converta dados brutos em vetores utilizando um modelo de embeddings (proprietário ou open source).
#     Armazene os vetores obtidos em uma VectorStore para facilitar a recuperação de informações.
#     Implemente a transformação de consultas dos usuários em vetores para buscar similaridade semântica.
#     Utilize uma métrica de similaridade, como a similaridade de cosseno, para comparar os vetores.
#     Estabeleça critérios para definir o uso de embeddings pagos ou open source, considerando a sensibilidade dos dados.
#     Teste o sistema com diferentes cenários, verificando a precisão na recuperação do contexto desejado.

from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from docling.document_converter import DocumentConverter


modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

converter = DocumentConverter()

result = converter.convert(Path('/home/guilherme/Desktop/langchain-aprendizagem/LANGCHAIN-APRENDIZAGEM/embeddings/Teen_Mental_Health_Dataset.csv'))

dados = result.document.export_to_markdown()

print(dados)

vetor_transformer = modelo_embedding.encode(dados)

print(vetor_transformer)