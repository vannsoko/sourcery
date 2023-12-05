from transformers import pipeline

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

with open('text.txt', 'r') as f:
    context = ' '.join(f.readlines())


result = question_answerer(question="resume the file", context=context)

print(f"Answer: '{result['answer']}',"
      f" score: {round(result['score'], 4)},"
      f" start: {result['start']},"
      f" end: {result['end']}")
