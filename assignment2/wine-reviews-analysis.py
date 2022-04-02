
# coding: utf-8

# # 作业2: 频繁模式与关联规则挖掘
# 
# 3120215523 金玉卿
# 
# 仓库地址 https://github.com/Reyna-Jin/DataMining/tree/main/assignment2

# ## 1.问题描述
# 选择1个数据集进行频繁模式和关联规则挖掘。
# 
# 数据分析要求
# - 对数据集进行处理，转换成适合进行关联规则挖掘的形式；
# - 找出频繁模式；
# - 导出关联规则，计算其支持度和置信度;
# - 对规则进行评价，可使用Lift、卡方和其它教材中提及的指标, 至少2种；
# - 对挖掘结果进行分析；
# - 可视化展示

# ### 数据集
# 
# wine-reviews
# 
# 一共2个csv文件
# 
# - winemag-data_first150k.csv
# 
# 包含10列和15万条葡萄酒评论
# 
# - winemag-data_first150k.csv
# 
# 包含10列和13万行葡萄酒评论
# 
# 这里我们首先分析winemag-data_first150k.csv文件的情况，数据属性如下列出：
# 
# - country 国家 
# - desprition 描述
# - designation 葡萄酒庄
# - pints 得分
# - price 价格
# - province 省份
# - region_1 区域1
# - region_2 区域2
# - variety 葡萄种类
# - winery 酿酒厂

# ## 2.数据处理

# 首先导入数据集合

# In[1]:

import matplotlib
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
path_15k = "../data/wine-reviews/winemag-data_first150k.csv"
data_15k = pd.read_csv(path_15k)


# 首先需要对数据集中的不同的属性进行处理
# 
# 1. 数据集中第一个属性未命名，是评论的序号，是唯一的，description属性是对于葡萄酒的自然语言描述，也是唯一值，二者在分析过程中不做考虑。

# 2. country、province、region_1和region_2是对葡萄酒产地的位置信息，出于分析复杂性和这四个属性的数据缺失情况考虑，这四个属性中只选择country进行挖掘。country属性中存在3个缺失值，所以需要通过属性的相关关系来填补缺失值，使用designation的属性来判断所属国家。

# In[2]:

#根据空值的分布，定义一个从designation到country的转换字典
designation2country = {
    "Askitikos":"Greece",    
    "Shah":"US",
    "Piedra Feliz":"Chile",
}
#处理country的空值
def country_nan_hander(data):
    for i in range(0,len(data)):
        tmp = data.iloc[i,1]
        if pd.isnull(tmp):
            designation = data.iloc[i,3]
            data.iloc[i,1] = designation2country[designation]
    return data


# 3. price、points是数值属性，对price进行离散化处理，此外points和price属性需要加上前缀，方便区分频繁项生成结果。
# 

# In[3]:

def points_discretization(value):
    return "points-"+str(int(value/5))

def price_discretization(value):
    if value < 100:
        return "price-"+str(int(value/10))
    else:
        return "price-10"


# 4. variety、winery、designation三个标称属性聚类数目过多（分别达到了632、14810、30622项），出于计算复杂度的考虑，在初步分析之后，单独选取选取variety中出现频数大于4000和winery中出现频数大于200的非空聚类进行分析。

# 初步分析过程中选取的属性包括designation、country、price、points，在之后的找出频繁模式调用mlxtend库来实现，因此还需要将数据处理成相应的格式。

# In[4]:

data_15k = pd.read_csv(path_15k)

#处理country的空值
country_nan_hander(data_15k)

#过滤属性
data_15k = data_15k.drop(['Unnamed: 0','description','province','region_1','region_2','variety','winery','designation'],axis = 1)


# In[5]:

#离散化处理
data_15k.loc[:,'points'] = data_15k['points'].map(lambda x:points_discretization(x))
data_15k.loc[:,'price'] = data_15k['price'].map(lambda x:price_discretization(x))


# In[6]:

#dataframe转换为列表
def deal(data):
    return data.to_list()
data_15k_arr = data_15k.apply(deal,axis=1).tolist()


# In[7]:

#TransactionEncoder转换
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
tf = te.fit_transform(data_15k_arr)
new_df = pd.DataFrame(tf,columns=te.columns_)


# ## 3.频繁模式

# 然后调用mlxtend中的apriori函数寻找频繁模式，最小支持度阈值取0.03

# In[8]:

from mlxtend.frequent_patterns import apriori
result = apriori(new_df, min_support=0.03, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)


# In[9]:

print(result.shape)
result[:20]


# ## 4.关联规则

# 然后从频繁项集中导出关联规则，并计算其支持度和置信度。这里使用mlxtend包中的association_rules方法，支持度阈值为0.03，置信度阈值设为0.4，方法默认状态下会计算关联规则的计算支持度、置信度和提升度。

# In[10]:

from mlxtend.frequent_patterns import association_rules
rules =  association_rules(result,metric ='confidence',min_threshold = 0.4)
rules = rules.drop(['leverage','conviction'],axis = 1)
print(rules.shape)
rules


# 如下列出导出的各项关联规则：

# In[11]:

for index, row in rules.iterrows():
    #print(row)
    t1 = tuple(row['antecedents'])
    t2 = tuple(row['consequents'])
    print("%s ⇒ %s (suupport = %f, confidence = %f )"%(t1,t2,row['support'],row['confidence']))


# ## 5.规则评价

# 然后对规则进行评价，这里使用提升度Lift和全置信度allconf。提升度Lift已经在 4.导出关联规则 的过程中被计算出来了，如下计算全置信度。

# In[12]:

def allconf(x):
    return x.support/max(x['antecedent support'],x['consequent support'])
allconf_list = []
for index, row in rules.iterrows():
    allconf_list.append(allconf(row))
rules['allconf'] = allconf_list
rules.drop(['antecedent support','consequent support'],axis=1,inplace=False)#.sort_values(by=['lift'], ascending=False)


# 过滤allconf小于0.1的规则，按照lift从大到小排序取前16项，得到用于分析的关联规则。

# In[13]:

final_rules = rules.iloc[:]
from sklearn.preprocessing import LabelEncoder
for index, row in final_rules.iterrows():
    #print(row)
    if row['allconf'] < 0.1:
        final_rules.drop(index=index,inplace=True) 
final_rules = final_rules.sort_values(by=['lift'], ascending=False)[:16]
final_rules


# ## 6.结果分析/可视化展示
# 

# 最后生成的规则如下列出：

# In[14]:

i = 1
for index, row in final_rules.iterrows():
    t1 = tuple(row['antecedents'])
    t2 = tuple(row['consequents'])
    print("%d : %s ⇒ %s (suupport = %f, confidence = %f )"%(i,t1,t2,row['support'],row['confidence']))
    i = i + 1


# - 在price和points的数值越大代表价格越高、分数越高。根据规则2，3，4，7，9，10，13可以看出，价格对葡萄酒的评分存在一定的影响，价格比较低（price-1和price-2，对应价格区间为10-29）的葡萄酒的评分更多地集中在16和17的评分档位（对应百分制评分的80-89）。而价格相对较高的葡萄酒（price-3到price-10，价格为30以上的）评分集中在18的评分档位（对应百分制评分的90-95），而且当价格高于price-40（price>40）档位后，评分并不会升高。
# 

# ![alt 属性文本](price-points.jpg)

# - 从('price-4',) ⇒ ('US',) ('price-2',) ⇒ ('US',) ('price-16',) ⇒ ('US',) ('price-1', 'US')的规则可以看出，来自美国的葡萄酒的价格分布比较广泛。

# - 从('price-10',) ⇒ ('France',),('Italy',) ⇒ ('points-17',)的规则可以看出，法国的葡萄酒的价格较高（price超过100），来自意大利的葡萄酒评分居中（points位于85-90之间）。

# ### 可视化展示

# 使用散点图可视化生成的rules规则：

# In[15]:

import  matplotlib.pyplot as plt
plt.xlabel('support')
plt.ylabel('confidence')
for i in range(rules.shape[0]):
    plt.scatter(rules.support[i],rules.confidence[i],c='r')


# In[16]:

plt.xlabel('support')
plt.ylabel('lift')
for i in range(rules.shape[0]):
    plt.scatter(rules.support[i],rules.lift[i],c='r')


# ## 考虑variety和winery属性的频繁模式与关联规则挖掘

# ### 数据处理

# In[17]:

df2 = pd.read_csv(path_15k)

#处理country的空值
country_nan_hander(df2)

#过滤属性
df2 = df2.drop(['Unnamed: 0','description','province','region_1','region_2','designation'],axis = 1)

#离散化处理
df2.loc[:,'points'] = df2['points'].map(lambda x:points_discretization(x))
df2.loc[:,'price'] = df2['price'].map(lambda x:price_discretization(x))


# In[32]:

#选取variety中出现频数大于4000的非空聚类所包括的行
variety_group = df2['variety'].value_counts()
variety_keys = []
for k in variety_group.keys():
    if variety_group[k]>4000: variety_keys.append(k)
df2_v = df2.loc[df2['variety'].isin(variety_keys)]
df2_v.drop(['winery'],axis = 1,inplace = True)



# In[33]:

#选取winery中出现频数大于200的非空聚类所包括的行
winery_group = df2['winery'].value_counts()
winery_keys = []
for k in winery_group.keys():
    if winery_group[k]>200: winery_keys.append(k)
df2_w = df2.loc[df2['winery'].isin(winery_keys)]
df2_w.drop(['variety'],axis = 1,inplace = True)


# In[19]:

#variety dataframe转换为列表
def deal(data):
    return data.to_list()
df2_v_arr = df2_v.apply(deal,axis=1).tolist()

#variety TransactionEncoder转换
te = TransactionEncoder()
tf = te.fit_transform(df2_v_arr)
new_df2_v = pd.DataFrame(tf,columns=te.columns_)


# In[20]:

#winery dataframe转换为列表
def deal(data):
    return data.to_list()
df2_w_arr = df2_w.apply(deal,axis=1).tolist()

#winery TransactionEncoder转换
te = TransactionEncoder()
tf = te.fit_transform(df2_w_arr)
new_df2_w = pd.DataFrame(tf,columns=te.columns_)


# ### 频繁模式

# variety和其它属性的频繁模式，最小支持度阈值取0.05

# In[21]:

variety_result = apriori(new_df2_v, min_support=0.05, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)


# In[22]:

variety_result


# winery和其它属性的频繁模式，最小支持度阈值取0.05

# In[23]:

winery_result = apriori(new_df2_w, min_support=0.05, use_colnames=True, max_len=4).sort_values(by='support', ascending=False)


# In[24]:

winery_result


# ### 导出关联规则/规则评价

# 然后从频繁项集中导出关联规则，并计算其支持度和置信度，支持度阈值为0.05，置信度阈值设为0.1，方法默认状态下会计算关联规则的计算支持度、置信度和提升度,此外额外计算规则的全置信度。

# In[25]:

#variety 关联规则导出
rules_v =  association_rules(variety_result,metric ='confidence',min_threshold = 0.5)
rules_v = rules_v.drop(['leverage','conviction'],axis = 1)

allconf_list = []
for index, row in rules_v.iterrows():
    allconf_list.append(allconf(row))
rules_v['allconf'] = allconf_list

print(rules_v.shape)
rules_v[:]


# In[26]:

#winery 关联规则导出
rules_w =  association_rules(winery_result,metric ='confidence',min_threshold = 0.5)
rules_w = rules_w.drop(['leverage','conviction'],axis = 1)

allconf_list = []
for index, row in rules_w.iterrows():
    allconf_list.append(allconf(row))
rules_w['allconf'] = allconf_list

print(rules_w.shape)


# ### 结果分析/可视化

# 在variety和其它属性（price、points和country）导出的关联规则中，列出提升度前20条规则。

# In[27]:

rules_v.sort_values(by='lift', ascending=False)[:20]


# - (Bordeaux-style Red Blend)→(France)	可以看出Bordeaux-style Red Blend品种的葡萄大都种植在法国
# - (Syrah)→(US)、 (Pinot Noir)→(US)、(Cabernet Sauvignon)→(US)  Syrah、Pinot Noir和Cabernet Sauvignon品种的葡萄大都种植在美国

# 在winery和其它属性（price、points和country）导出的关联规则中，列出提升度前30条规则。

# In[28]:

rules_w.sort_values(by='lift', ascending=False)[:12]


# - (France)→(Bouchard Père & Fils)，Bouchard Père & Fils是法国比较普遍的葡萄酒庄园
# - (Trapiche)→(Argentina)和(Argentina)→(Trapiche)	Trapiche是阿根廷比较普遍的葡萄酒庄园
# - (Trapiche)→(Argentina, price-1) Trapiche葡萄酒庄园的葡萄酒价格较为便宜(价格区间在11-19之间)

# variety和其它属性关联规则可视化

# In[29]:

import  matplotlib.pyplot as plt
plt.xlabel('support')
plt.ylabel('confidence')
for i in range(rules_v.shape[0]):
    plt.scatter(rules_v.support[i],rules_v.confidence[i],c='r')


# winery和其它属性关联规则可视化

# In[30]:

import  matplotlib.pyplot as plt
plt.xlabel('support')
plt.ylabel('confidence')
for i in range(rules_w.shape[0]):
    plt.scatter(rules_w.support[i],rules_w.confidence[i],c='r')

