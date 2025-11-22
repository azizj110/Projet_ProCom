import pandas as pd
from main import CACHE_DIR

higher_categories_path=CACHE_DIR/"category53_wide-format.tsv"
df = pd.read_csv(higher_categories_path, sep="\t")
# print(df.head())

category_sum = df.iloc[:, 2:].sum(axis=1)
objects_five_cat = df.loc[category_sum >= 2, "uniqueID"]
print(len(objects_five_cat.tolist()))

cat_cols = df.columns[2:]  
for object in objects_five_cat.tolist():
    row = df.loc[df["uniqueID"] == object, cat_cols]
    categories = list(cat_cols[row.values[0] == 1])
    print(f"{object} appartient aux catégories : {categories}")