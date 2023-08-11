

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.api.types import EmbeddingFunction
import traceback as tb
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import json, uvicorn

model_id = ModelTypes.FLAN_UL2
api_key = 'V_cAKE4oOEUjejf352EZCsNJa-rF_FGDcfVzTYEwtzFg'
project_id = '71abf865-d056-419f-b21b-1903d677cdae'
model_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": 'ZFRVJAjxWLVPux5bq4ylcpyaH-hMJtNgSNuGMV8QmUkl'
}

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()


class ChromaWithUpsert:
    def __init__(
            self,
            name = "watsonx_rag_collection",
            persist_directory='data/knowledge_base',
            embedding_function=None,
            collection_metadata = None,
    ):
        self._client_settings = chromadb.config.Settings()
        if persist_directory is not None:
            self._client_settings = chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
            )
        self._client = chromadb.Client(self._client_settings)
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._name = name
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            embedding_function=self._embedding_function
            if self._embedding_function is not None
            else None,
            metadata=collection_metadata,
        )

    def upsert_texts(
        self,
        texts,
        metadata = None,
        ids = None,
        **kwargs,
    ):
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            :param texts (Iterable[str]): Texts to add to the vectorstore.
            :param metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            :param ids (Optional[List[str]], optional): Optional list of IDs.
            :param metadata: Optional[List[dict]] - optional metadata (such as title, etc.)
        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(0, len(texts))]
        embeddings = None
        self._collection.upsert(
            metadatas=metadata, documents=texts, ids=ids
        )
        return ids

    def is_empty(self):
        return self._collection.count()==0

    def persist(self):
        self._client.persist()

    def query(self, query_texts:str, n_results:int=5):
        """
        Returns the closests vector to the question vector
        :param query_texts: the question
        :param n_results: number of results to generate
        :return: the closest result to the given question
        """
        return self._collection.query(query_texts=query_texts, n_results=n_results)


def make_prompt(context, question_text):
    return (f"Please answer the following.\n"
          + f"{context}:\n\n"
          + f"{question_text}")

def make_prompts(relevant_contexts, question_texts):
    prompt_texts = []
    for relevant_context, question_text in zip(relevant_contexts, question_texts):
        context = "\n\n\n".join(relevant_context["documents"][0])
        prompt_text = make_prompt(context, question_text)
        prompt_texts.append(prompt_text)
    return prompt_texts


def create_model():
    parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.MAX_NEW_TOKENS: 100
    }
    return Model(model_id=model_id, params=parameters, credentials=model_credentials, project_id=project_id)


def get_relevant_embeddings(question_texts):
    relevant_contexts = []
    emb_func = MiniLML6V2EmbeddingFunction()
    chroma = ChromaWithUpsert(name=f"nq910_minilm6v2", embedding_function=emb_func,)
    for question_text in question_texts:
        relevant_chunks = chroma.query(
            query_texts=[question_text],
            n_results=2,
        )
        relevant_contexts.append(relevant_chunks)
    return relevant_contexts
    
model = create_model()

def get_answers(question_texts):
    results = []
    try:
        relevant_contexts = get_relevant_embeddings(question_texts)
        prompt_texts = make_prompts(relevant_contexts, question_texts)
        for prompt_text in prompt_texts:
            results.append(model.generate_text(prompt=prompt_text))
    except:
        tb.print_exc()
    return results


class Query(BaseModel):
    query: str = Field(...)


@app.post("/chat/")
def query_bot(query: Query = Body(...)) -> dict:
    req = query.dict()['query']
    resp = get_answers(req)
    return {'response': resp}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)