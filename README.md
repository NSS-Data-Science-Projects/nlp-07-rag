## Retrieval-Augmented Generation

In this exercise, you'll put together a RAG system and compare outputs from RAG vs. just querying an LLM.

For this exercise, you'll be asking about Subspace-Constrained LoRA (SC-LoRA), a new technique described in [a recent article publised on arXiv.org](https://arxiv.org/abs/2505.23724). You've been provided the text of this article in the file 2505.23724v1.txt.

First, you'll need to set up a way to interact with the generator model. You can use the OpenAI class from the openai library for this. See [this page](https://developers.openai.com/api/docs/guides/text?lang=python) for more information. When you do this, you'll need to set the base_url to "https://openrouter.ai/api/v1" and to pass in your api key. Set the model to "openrouter/owl-alpha".

First, ask the model "How does SC-LoRA differ from regular LoRA?" without providing any additional context. Read through a few different responses. **Question:** Are the responses accurate, or does it seem like the model is just making up something that sounds plausible?


### Part 1: Manual RAG

In order to get more reliable responses, let's set up a RAG system.

In this first part, you'll build all of the pieces of the RAG system individually.

First, you'll need the retriever portion. Create a FAISS index to hold the text of the article. Encode this text using the all-MiniLM-L6-v2 encoder. Note that you'll want to divide the text into smaller chunks rather than encoding the whole article all at once. You could try, for example, the [RecursiveCharacterTextSplitter class from LangChain](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter). You'll need to specify a chunk_size and chunk_overlap. You could try a chunk_size of 500 and overlap of 50 as a starting point.

Next, use the following as a system prompt:

```
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    f"Context: {context}"
)
```

Use the FAISS index to pull in relevant context to fill in the context. Try passing in this additional system prompt. Hint: you can do this by using the following messages in the client.chat.completions.create function

```
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": query,
        }
    ]
```

How does adding this context change the results?

### Part 2: LangChain

You can also use the [LangChain library](https://www.langchain.com/) to help build your RAG system.

For the retriever, you can use the [HugginFaceEmbeddings class](https://reference.langchain.com/python/langchain-huggingface/embeddings/huggingface/HuggingFaceEmbeddings), using the all-MiniLM-L6-v2 model, to create your embedding model. There is also a [FAISS class](https://reference.langchain.com/python/langchain-community/vectorstores/faiss/FAISS), which has a useful from_texts method. Once you've created your vector store, use the [as_retriever method](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever) on it and save it to a variable named `retriever`.

For the generator, you can use the [ChatOpenAI class](https://python.langchain.com/docs/integrations/chat/openai/). Be sure to set base_url="https://openrouter.ai/api/v1", model_name="openrouter/owl-alpha", and openai_api_key= Your API key. Save this to a variable named `llm`.


Now that the two components have been created, we can combine them into a chat template using the [ChatPromptTemplate class](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html). We can set up a system prompt and the pass that in, like
```
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
```

Now we need to connect all of the pieces together. Newer versions of LangChain commonly use LCEL (LangChain Expression Language)[https://www.langchain.com/blog/langchain-expression-language] to build pipelines where components are connected together using the | operator.

You'll need:

* A helper function that combines retrieved documents into a single string of context
* A pipeline that:
    - extracts the question from the input
    - sends the question to the retriever
    - formats the retrieved documents into context
    - passes both the context and question into the prompt
    - sends the completed prompt to the LLM

A simplified diagram looks like this:

input
   ↓
extract question
   ↓
retrieve documents
   ↓
format context
   ↓
prompt
   ↓
LLM

You can create this chain by using 

```
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

rag_chain = (
    {
        "context": (
            RunnableLambda(lambda x: x["question"])
            | retriever
            | RunnableLambda(format_docs)
        ),
        "question": RunnableLambda(
            lambda x: x["question"]
        )
    }
    | prompt
    | llm
)
```

Take a minute to study this and see if you can figure out how the syntax works.

Finally, invoke your chain using:

```
response = rag_chain.invoke(
    {"question": query}
)

print(response.content)
```

Compare the output from this section with both previous approaches:
* LLM without retrieval
* Manual RAG
* LangChain RAG

The quality of the answers from (2) and (3) should be similar. The purpose of LangChain is not to improve the model itself. Rather, it provides abstractions that simplify the retrieval and generation workflow.