# ==========================================
# Titanic Kaggle Competition 
# ==========================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import re



# =============== 参数区：请在这里修改您的文件路径 ======================
TRAIN_DATA_PATH = r"C:\Users\Administrator\Desktop\KAGGLE\train.csv"
TEST_DATA_PATH  = r"C:\Users\Administrator\Desktop\KAGGLE\test.csv"
# ========================================================================

# =============== 第一步：读取数据 ======================
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# =============== 第二步：特征工程 ======================

# 1. 缺失值填充
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age']  = test_df['Age'].fillna(train_df['Age'].median())

train_df['Embarked'] = train_df['Embarked'].fillna('S')
test_df['Embarked']  = test_df['Embarked'].fillna('S')

test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())

# 2. 转换性别为数值
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex']  = test_df['Sex'].map({'male': 0, 'female': 1})

# 3. 处理登船港口：数值映射
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train_df['Embarked'] = train_df['Embarked'].map(embarked_mapping)
test_df['Embarked']  = test_df['Embarked'].map(embarked_mapping)

# 4. 新增简单特征：FamilySize = SibSp + Parch + 1
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize']  = test_df['SibSp'] + test_df['Parch'] + 1

# 4.1 定义是否单独出行
train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
test_df['IsAlone']  = (test_df['FamilySize'] == 1).astype(int)

# 5. 从姓名中提取头衔 (Title)
def get_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

for df in [train_df, test_df]:
    df['Title'] = df['Name'].apply(get_title)
    # 将稀有头衔合并为 'Rare'
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev',
                                       'Sir','Jonkheer','Dona'],'Rare')
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')

# 用数字映射 Title
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for df in [train_df, test_df]:
    df['Title'] = df['Title'].map(title_mapping).fillna(0).astype(int)

# =============== 选取最终用于训练的特征列 ======================
# 原有基础特征 + 新增 'IsAlone' + 'Title'
feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                'FamilySize', 'IsAlone', 'Title']

X = train_df[feature_cols]
y = train_df['Survived']
X_test = test_df[feature_cols]

# =============== 第三步：本地交叉验证 ======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# LightGBM 参数
params = {

    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.0125,
    'num_leaves': 31,
    'feature_fraction': 0.8,   # 随机采样80%特征
    'bagging_fraction': 0.8,   # 随机采样80%数据
    'bagging_freq': 5,         # 每5轮迭代一次bagging
    'seed': 42,
    'verbose': -1
}

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2100,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train','valid'],
        #early_stopping_rounds=50,  # 如果在验证集上50轮没有提升，就停止
        #verbose_eval=False
    )

    # 预测验证集
    val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.where(val_pred_prob > 0.5, 1, 0)

    acc = accuracy_score(y_val, val_pred)
    cv_scores.append(acc)

print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average CV Accuracy: {np.mean(cv_scores):.4f}")

# # =============== 第四步：在全量训练集上重新训练 ======================
# final_lgb_train = lgb.Dataset(X, y)
# final_model = lgb.train(
#     params,
#     final_lgb_train,
#     num_boost_round=500,
#     #verbose_eval=False
# )

# =============== 第五步：对测试集进行预测并生成提交文件 ======================
test_pred_prob = final_model.predict(X_test)
test_pred = np.where(test_pred_prob > 0.5, 1, 0)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": test_pred
})

submission.to_csv("submission.csv", index=False)
print("=== Done! 'submission.csv' has been generated. ===")

