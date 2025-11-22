import pandas as pd

# Lire les fichiers
df = pd.read_csv("./data/category53_wide-format.tsv", sep="\t")
resolved_df = pd.read_csv("./data/resolved.csv")
resolved_df.columns = ['uniqueID', 'higher_category']

# Identifier les colonnes de catégorie
category_cols = df.columns[2:].tolist()

# Identifier les colonnes de catégorie
category_cols = df.columns[2:].tolist()

# 1. Fusionner les catégories résolues dans le DataFrame principal
df_merged = df.merge(
    resolved_df[['uniqueID', 'higher_category']],
    on='uniqueID',
    how='left'
)

# Identifier les lignes à mettre à jour (celles qui étaient dans resolved.csv)
rows_to_update = df_merged['higher_category'].notna()

# 2. Mettre à jour les colonnes de catégorie pour les lignes identifiées

# a. Mettre toutes les catégories originales à 0 pour les lignes à mettre à jour.
df_merged.loc[rows_to_update, category_cols] = 0

# b. Pour chaque ligne à mettre à jour, mettre la catégorie spécifique 'higher_category' à 1.
for index in df_merged[rows_to_update].index:
    new_cat = df_merged.loc[index, 'higher_category']
    # Vérifier si la catégorie cible est une colonne valide avant d'assigner
    if new_cat in category_cols:
        df_merged.loc[index, new_cat] = 1

# 3. Nettoyage final et sauvegarde

# Supprimer la colonne temporaire 'higher_category'
df_final = df_merged.drop(columns=['higher_category'])

# Enregistrer le fichier final
output_filename = "mapping_higher_categories.tsv"
df_final.to_csv(output_filename, sep="\t", index=False)

