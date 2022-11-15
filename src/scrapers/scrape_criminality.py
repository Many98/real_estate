import pandas as pd
'''
for i in range(12, 23):
    url = "https://kriminalita.policie.cz/api/v2/downloads/20"+str(i)+"_554782.geojson"
    url = requests.get(url)
    text = url.text
    print(i)
    with open('criminality_'+str(i)+'.geojson', 'a', encoding='utf-8', errors='ignore') as f:
        dump(text, f)
'''
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


df_crim = pd.DataFrame()
for j in range(12, 23):
    df = pd.read_csv('20'+str(j)+'_554782.csv') # locally too big
    df = df.drop(columns=['mp', 'state', 'relevance'])
    indexes = [13, 14, 15, 16, 17, 54, 55, 56, 57, 58, 59, 60, 61, 62, 74, 75, 76, 77, 78, 80, 81, 82, 83]
    indexes_2 = []
    for i in range(36):
        indexes_2.append(97+i)
    df = filter_rows_by_values(df, "types", indexes)
    df = filter_rows_by_values(df, "types", indexes_2)
    df_crim = df_crim.append(df)

# df_crim["types"] = df_crim["types"].astype(str)
df_crim["types"] = df_crim["types"].replace(79, "Dopravní nehody", regex=True)
for l in range(88, 97):
    df_crim["types"] = df_crim["types"].replace(l, "Extremismus", regex=True)
for m in range(63, 74):
    df_crim["types"] = df_crim["types"].replace(m, "Obecně nebezpečná", regex=True)
for k in range(35, 54):
    df_crim["types"] = df_crim["types"].replace(k, "Krádeže", regex=True)
for j in range(18, 35):
    df_crim["types"] = df_crim["types"].replace(j, "Krádeže vloupáním", regex=True)
for i in range(1, 13):
    df_crim["types"] = df_crim["types"].replace(i, "Násilné", regex=True)
print(df_crim.info())
df_crim.to_csv('criminality.csv', sep=',', encoding='utf-8')
