## Retrieval-Augmented Generation

In this exercise, you'll put together a RAG system and compare outputs from RAG vs. just querying an LLM.

For this exercise, you'll be asking about Subspace-Constrained LoRA (SC-LoRA), a new technique described in [a recent article publised on arXiv.org](https://arxiv.org/abs/2505.23724). You've been provided the text of this article in the file 2505.23724v1.txt.

### Part 1: Manual RAG

In this first part, you'll build all of the pieces of the RAG system individually.

First, you'll need the retriever portion. Create a FAISS index to hold the text of the article. Encode this text using the all-MiniLM-L6-v2 encoder. Note that you'll want to divide the text into smaller chunks rather than encoding the whole artile all at once. You could try, for example, the [RecursiveCharacterTextSplitter class from LangChain](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html). You'll need to specify a chunk_size and chunk_overlap. You could try a chunk_size of 500 and overlap of 50 as a starting point.

Next, you'll need to set up a way to interact with the generator model. You can use the OpenAI class from the openai library for this. See [this page](https://platform.openai.com/docs/api-reference/chat/create) for more information. When you do this, you'll need to set the base_url to "https://openrouter.ai/api/v1" and to pass in your api key. Set the model to "meta-llama/llama-4-scout:free".

First, ask the model "How does SC-LoRA differ from regular LoRA?" without providing any additional context. Read through a few different responses.

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

For the retriever, you can use the [HugginFaceEmbeddings class](https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html), using the all-MiniLM-L6-v2 model, to create your embedding model. There is also a [FAISS class](https://python.langchain.com/docs/integrations/vectorstores/faiss/), which has a useful [from_texts method](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.from_texts). Once you've created your vector store, use the [as_retriever method](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.as_retriever) on it and save it to a variable named `retriever`.

For the generator, you can use the [ChatOpenAI class](https://python.langchain.com/docs/integrations/chat/openai/). Be sure to set base_url="https://openrouter.ai/api/v1", model_name="meta-llama/llama-4-scout:free", and openai_api_key= Your API key. Save this to a variable named `llm`.

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

Then, you can use the [create_stuff_documents_chain function](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html), passing in your llm and the prompt, and then create a chain using the [create_retrieval_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html) function, passing in the retriever and the chain you just created.

Finally, you can use the invoke method to pass in your query as input. See the example on [this page](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval.create_retrieval_chain.html).