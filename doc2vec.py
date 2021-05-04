#Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

## Exapmple document (list of sentences)
doc = ["I love data science",
        "I love coding in python",
        "I love building NLP tool",
        "This is a good phone",
        "This is a good TV",
        "This is a good laptop"]


# 1.- Tokenización de cada documento, convirtiendo todo a palabras minusculas
tokenized_doc = []
for d in doc:
    tokenized_doc.append(word_tokenize(d.lower()))
tokenized_doc
#print(tokenized_doc)


# Convertir documento tokenizado en datos etiquetados con formato gensim 
# tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(doc)]
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
#tagged_data
#print(tagged_data)


##  Entrena el modelo doc2vec model     
# model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=2, workers=4, epochs = 100)
# alpha valor crítico
max_epochs = 100 
vec_size = 20 
alpha = 0.025 

model = Doc2Vec (vector_size = vec_size, alpha = alpha, min_alpha = 0.00025 , min_count = 1, dm = 1) 
model.build_vocab (tagged_data) 

for epoch in range(max_epochs):
    # print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=10)
    # disminuir la tasa de aprendizaje
    model.alpha -= 0.0002
    # corregir la tasa de aprendizaje, sin modelo de decadencia
    model.min_alpha = model.alpha

model.save("d2v.model")
print("\n",tagged_data,"\n")
print("\n-----------------------Modelo Guardado---------------------------------\n")

# dm define el algoritmo de tr a ining. Si dm = 1 significa 'memoria distribuida' (PV-DM) y 
# dm = 0 significa ' bolsa distribuida de palabras' (PV-DBOW). El modelo de memoria distribuida 
# conserva el orden de las palabras en un documento, mientras que la bolsa de palabras distribuida 
# solo utiliza el enfoque de la bolsa de palabras, que no conserva ningún orden de palabras.

# vector_size dimensión de neuronas de la capa oculta y del vector de salida
# Window Máxima distancia entre la palabra actual y la pronosticada dentro de una frase
# min_count Ignora palabras que aparecen menos que el valor configurado
# workers número de trheads para entrenar el modelo
# epochs número de iteraciones sobre el corpus

model= Doc2Vec.load("d2v.model")

# Guarda el modelo 
#model.save("test_doc2vec.model")
## Carga guardada doc2vec model 
#model= Doc2Vec.load("test_doc2vec.model")
## Vocabulario del modelo de impresión 
#print(model.wv.vocab)


# encontrar el documento más similar 
frase = "this is a good laptop"
print(frase,"\n")
test_doc = word_tokenize(frase.lower())
print(model.docvecs.most_similar(positive=[model.infer_vector(test_doc)],topn=1),"\n")

