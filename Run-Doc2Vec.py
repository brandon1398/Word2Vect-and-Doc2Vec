from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import gensim.models as g



texto = open('C:/Users/User/Documents/7TO CICLO/Inteligencia Artificial/Articulos/Practicas-Doc2Vec/Doc2Vec/DataSet/prueba.txt', encoding='utf-8') 
tokenized_doc = []

# Tokenización de cada documento, convirtiendo todo a palabras minusculas
def tokenizacion(line):
    for d in line:
        tokenized_doc.append(word_tokenize(d.lower()))
        
    
for line in texto:
    line = line.strip()
    line = line.split("\n")
    tokenizacion(line)


print("\nTokenización de cada documento, convirtiendo todo a palabras minusculas\n\n",tokenized_doc,"\n-------------------------------------------------\n")

# Convertir documento tokenizado en datos etiquetados con formato gensim 
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]     
print("\nConvertir documento tokenizado en datos etiquetados con formato gensim\n\n",tagged_data,"\n-------------------------------------------------\n")

max_epochs = 100 
vec_size = 20 
alpha = 0.025 

# Entrena el modelo doc2vec model 
model = Doc2Vec (vector_size = vec_size, alpha = alpha, min_alpha = 0.00025 , min_count = 1, dm = 1) 
model.build_vocab (tagged_data) 

for epoch in range(max_epochs):
    # print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=10)
    # disminuir la tasa de aprendizaje
    model.alpha -= 0.0002
    # corregir la tasa de aprendizaje, sin modelo de decadencia
    model.min_alpha = model.alpha

# Guarda el modelo 
model.save("d2v.model")
# Carga el model 
model= Doc2Vec.load("d2v.model")
# print("\nVocabulario del modelo de impresion\n\n",model.wv.vocab,"\n-------------------------------------------------\n")

print(model.wv.vocab)

frase1 = "ayuda casos trilogía"
frase2 = "recupere cuidado estafadores"
frase3 = "durante observas piedras sitios"

print("\n",frase1,"\n")
test_doc = word_tokenize(frase1.lower())
print(model.docvecs.most_similar(positive=[model.infer_vector(test_doc)],topn=1),"\n")


