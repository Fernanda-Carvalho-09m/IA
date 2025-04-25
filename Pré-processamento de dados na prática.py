# NAO ALTERE OS COMANDOS ABAIXO
import gdown
import pandas as pd
from IPython.display import display

dados_funcionarios = pd.read_excel('C:/Users/User/Desktop/IA - Pre-processamento de dados/MFGEmployees4.xlsx')

# Removendo registros duplicados na base
display (dados_funcionarios)
dados_funcionarios.duplicated().any()
dados_funcionarios.duplicated().sum()
dados_funcionarios[dados_funcionarios.duplicated()]
df_dados_funcionarios = dados_funcionarios.drop_duplicates()
display (df_dados_funcionarios)


# Removendo valores ausentes na base
df_dados_funcionarios.isna().sum()
df_dados_funcionarios = dados_funcionarios.dropna()
display (df_dados_funcionarios)


# Removendo os "rotulos"
dados_funcionarios.info()
df_dados_funcionarios = df_dados_funcionarios.drop(columns=['id','sobrenome', 'nome'])
df_dados_funcionarios.columns


# A idade mínima de um funcionário é 14 anos e máxima é de 80 anos.
# A duração do serviço não pode ultrapassar 44 horas.
print(df_dados_funcionarios.describe())
df_dados_funcionarios = df_dados_funcionarios[(df_dados_funcionarios['idade'] >= 14) &
                                              (df_dados_funcionarios['idade'] <= 80)]
df_dados_funcionarios = df_dados_funcionarios[df_dados_funcionarios['duracao_servico'] <= 44]
print(df_dados_funcionarios.describe())



#Separando lista com os nomes
alvo = "horas_ausencia"
atributos = ["sexo", "cidade", "cargo", "localizacao", "divisao", "idade", "duracao_servico", "unidade"]


#Criando bases separadas
Y = df_dados_funcionarios[alvo].to_frame()
X = df_dados_funcionarios[atributos]

#Separando os atributos em 2 classes
atributos_numericos = ["idade", "duracao_servico"]
atributos_categoricos = ["sexo", "cidade", "cargo", "localizacao", "divisao", "unidade"]


#Importanto bibliotecas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.decomposition import PCA
#Encadeando as 3 transformacoes para atributos numericos
num_pipe = Pipeline([('padronizacao', StandardScaler()),('transformacao', PowerTransformer()), ('componentes', PCA())])
#Sequenciando as transformacoes necessarias - transformacoes atributos numericos e atributos categoricos
atributos_transf = ColumnTransformer([('atributos_numericos', num_pipe, atributos_numericos), ('atributos_categoricos', OneHotEncoder(dtype = 'int', drop = 'first'), atributos_categoricos)],remainder='drop',verbose_feature_names_out=False)
#Aplicando internamente a transformacao
atributos_transf.fit(X,Y)
#Tendo uma ideia de como fica o resultado
display (pd.DataFrame(atributos_transf.transform(X).toarray(), columns=atributos_transf.get_feature_names_out()).head())
#Sequenciando as transformacoes necessarias para um alvo numerico
alvo_num = TransformedTargetRegressor(regressor=None, transformer=num_pipe)
#Sequenciando as transformacoes necessarias para um alvo categorico
alvo_cat = Pipeline(steps = [('codificacao', OneHotEncoder(dtype = 'int', drop = 'first'))])
#Pipeline Alvo Numerico
pipe_num = Pipeline(steps = [('atributos', atributos_transf), ('alvo', alvo_num)])
display (pipe_num)
#Pipeline Alvo Categorico
pipe_cat = Pipeline(steps = [('atributos', atributos_transf), ('alvo', alvo_cat)])
display (pipe_cat)