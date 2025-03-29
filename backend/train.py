import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset
avaliacoes = [
    "O atendimento foi excelente!",
    "Demorou muito para resolver meu problema.",
    "O atendimento foi bom, mas poderia ser melhor.",
    "Adorei a agilidade no suporte.",
    "Fiquei insatisfeito com a demora na resposta.",
    "O serviço foi eficiente, mas o atendente poderia ser mais simpático.",
    "Resolução rápida e eficaz, recomendo!",
    "Não gostei da falta de comunicação durante o processo.",
    "Atendimento cordial, mas o problema não foi totalmente resolvido.",
    "Muito satisfeito com o suporte prestado.",
    "Demoraram para atender minha solicitação, mas no final resolveram.",
    "O atendente foi muito prestativo e solucionou meu problema rapidamente.",
    "Achei o atendimento um pouco frio e impessoal.",
    "Excelente experiência, resolveram tudo em poucos minutos.",
    "Fiquei frustrado com a falta de profissionalismo.",
    "O atendimento foi razoável, mas esperava mais agilidade.",
    "Superou minhas expectativas, muito obrigado!",
    "Não recomendo, o serviço foi péssimo.",
    "Atendimento mediano, nada excepcional.",
    "Resolução rápida, mas o atendente poderia ser mais atencioso.",
    "Fiquei muito satisfeito com a qualidade do atendimento.",
    "Demorou muito para me atenderem, e ainda não resolveram meu problema.",
    "O atendimento foi bom, mas o processo foi muito burocrático.",
    "Adorei a simpatia do atendente, mas o problema ainda persiste.",
    "Não gostei da forma como fui tratado.",
    "O suporte foi eficiente e resolveu meu problema rapidamente.",
    "Achei o atendimento muito lento e desorganizado.",
    "Fiquei impressionado com a rapidez e eficiência.",
    "O atendente foi muito gentil, mas a solução demorou mais do que o esperado.",
    "Não fui bem atendido, senti falta de empatia.",
    "O atendimento foi ótimo, mas o problema poderia ter sido resolvido mais rápido.",
    "Fiquei muito feliz com a solução apresentada.",
    "Demoraram muito para me atender, e o problema ainda não foi resolvido.",
    "O atendimento foi satisfatório, mas poderia ser mais ágil.",
    "Adorei a atenção e o cuidado do atendente.",
    "Não gostei da demora e da falta de informações claras.",
    "O suporte foi excelente, resolveram tudo rapidamente.",
    "Achei o atendimento muito ruim, não recomendo.",
    "Fiquei satisfeito com a solução, mas o processo foi demorado.",
    "O atendente foi muito atencioso e resolveu meu problema de forma eficiente.",
    "Não fui bem atendido, senti que meu problema não foi prioridade.",
    "O atendimento foi bom, mas a comunicação poderia ser melhor.",
    "Adorei a rapidez e a eficiência do suporte.",
    "Fiquei insatisfeito com a falta de profissionalismo do atendente.",
    "O atendimento foi mediano, nada demais.",
    "Resolução rápida e eficiente, recomendo a todos.",
    "Não gostei da forma como meu problema foi tratado.",
    "O atendente foi muito simpático, mas a solução demorou mais do que o esperado.",
    "Fiquei muito satisfeito com o atendimento e a solução apresentada.",
    "Demoraram muito para resolver meu problema, mas no final deu tudo certo."
]

sentimentos = [
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "neutro",
    "positivo",
    "negativo",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "neutro",
    "negativo",
    "positivo",
    "negativo",
    "positivo",
    "neutro",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "negativo",
    "neutro",
    "positivo",
    "neutro"
]

data = {
    "Avaliações": avaliacoes,
     "Sentimento": sentimentos
}
df = pd.DataFrame(data)
print(df.head())

x = df["Avaliações"]
y = df["Sentimento"]

# Vetorização do texto
vectorizer = TfidfVectorizer(stop_words=None, max_df=0.95, min_df=2, ngram_range=(1, 2))
x_vectorized = vectorizer.fit_transform(x)

# Balanceamento de classes
ros = RandomOverSampler(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x_vectorized, y)

# Divisão treino/teste
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42)

# Treinamento do modelo Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)

# Predições
y_pred = model.predict(x_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negativo", "neutro", "positivo"], yticklabels=["negativo", "neutro", "positivo"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Salvar modelo e vetorizer
joblib.dump(model, "../sentiment_model.pkl")
joblib.dump(vectorizer, "../vectorizer.pkl")
print("Modelo e vetorizer salvos com sucesso!")
