from utils.llm import LLM
from utils.build_rag import RAG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def predict_rag(qns: str, history=None) -> str:
    #llm = LLM().get_llm_together()
    llm = LLM().openai()
    retriever = RAG().get_retriever()

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    retrieval_chain = (
        {
            "context": retriever | RunnableLambda(
                lambda docs: "\n".join([d.page_content for d in docs])
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retrieval_chain.invoke(qns)