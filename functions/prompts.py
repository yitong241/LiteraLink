prompt_templates = {
    'summarization': "Write a short summary for this text: {text}",
    'question-answering': "Read this and answer the question. If the question is unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}"
}

qa_prompts = [
    ("Please answer a question about the following article about "
     "{title}:\n\n{context}\n\n{question}", "{answer}"),
    ("Read this and answer the question\n\n{context}\n\n{question}",
     "{answer}"),
    ("{context}\n{question}", "{answer}"),
    ("Answer a question about this article:\n{context}\n{question}",
     "{answer}"),
    ("Here is a question about this article: {context}\nWhat is the answer"
     " to this question: {question}", "{answer}"),
    ("Article: {context}\n\nQuestion: {question}", "{answer}"),
    ("Article: {context}\n\nNow answer this question: {question}",
     "{answer}"),
    ("{title}\n{context}\n\nQ: {question}", "{answer}"),
    ("Ask a question about {title}.", "{question}"),
    ("What is the title of this article:\n\n{context}", "{title}"),
    ("{title}:\n\n{context}\n\nPlease answer a question about this "
     "article. If the question is unanswerable, say \"unanswerable\". "
     "{question}", "{answer}"),
    ("Read this and answer the question. If the question is unanswerable, "
     "say \"unanswerable\".\n\n{context}\n\n{question}", "{answer}"),
    ("What is a question about this article? If the question is "
     "unanswerable, say \"unanswerable\".\n\n{context}\n\n{question}",
     "{answer}"),
    ("{context}\n{question} (If the question is unanswerable, say "
     "\"unanswerable\")", "{answer}"),
    ("{context}\nTry to answer this question if possible (otherwise reply "
     "\"unanswerable\"): {question}", "{answer}"),
    ("{context}\nIf it is possible to answer this question, answer it for "
     "me (else, reply \"unanswerable\"): {question}", "{answer}"),
    ("{context}\n\nAnswer this question, if possible (if impossible, reply"
     " \"unanswerable\"): {question}", "{answer}"),
    ("Read this: {context}\n\n{question}\nWhat is the answer? (If it "
     "cannot be answered, return \"unanswerable\")", "{answer}"),
    ("Read this: {context}\nNow answer this question, if there is an "
     "answer (If it cannot be answered, return \"unanswerable\"): "
     "{question}", "{answer}"),
    ("{context}\nIs there an answer to this question (If it cannot be "
     "answered, say \"unanswerable\"): {question}", "{answer}"),
]

summarization_prompts = [
    ("Write highlights for this article:\n\n{text}", "{highlights}"),
    ("Write some highlights for the following article:\n\n{text}",
     "{highlights}"),
    ("{text}\n\nWrite highlights for this article.", "{highlights}"),
    ("{text}\n\nWhat are highlight points for this article?",
     "{highlights}"),
    ("{text}\nSummarize the highlights of this article.", "{highlights}"),
    ("{text}\nWhat are the important parts of this article?",
     "{highlights}"),
    ("{text}\nHere is a summary of the highlights for this article:",
     "{highlights}"),
    ("Write an article using the following points:\n\n{highlights}",
     "{text}"),
    ("Use the following highlights to write an article:\n\n{highlights}",
     "{text}"),
    ("{highlights}\n\nWrite an article based on these highlights.",
     "{text}")
]
