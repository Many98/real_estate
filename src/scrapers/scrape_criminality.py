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
    indexes = [1, 13, 14, 15, 16, 17, 18, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 70, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84]
    indexes_2 = []
    indexes_3 = []
    for i in range(25):
        indexes_2.append(44+i)
    for i in range(44):
        indexes_2.append(89+i)
    df = filter_rows_by_values(df, "types", indexes)
    df = filter_rows_by_values(df, "types", indexes_2)
    df = filter_rows_by_values(df, "types", indexes_3)
    df = df.reset_index(drop=True)
    df_crim = df_crim.append(df)

df_crim["types"] = df_crim["types"].replace(2, "vražda", regex=True)
df_crim["types"] = df_crim["types"].replace(3, "loupež", regex=True)
df_crim["types"] = df_crim["types"].replace(4, "vydírání", regex=True)
df_crim["types"] = df_crim["types"].replace(5, "výtržnictví", regex=True)
df_crim["types"] = df_crim["types"].replace(6, "nebezpečné vyhrožování", regex=True)
df_crim["types"] = df_crim["types"].replace(7, "rvačka", regex=True)
df_crim["types"] = df_crim["types"].replace(8, "úmyslné ublížení na zdraví", regex=True)
df_crim["types"] = df_crim["types"].replace(9, "útok proti výkonu pravomoci stát. orgánu", regex=True)
df_crim["types"] = df_crim["types"].replace(10, "omezování osobní svobody", regex=True)
df_crim["types"] = df_crim["types"].replace(11, "únos", regex=True)
df_crim["types"] = df_crim["types"].replace(12, "obchod s lidmi", regex=True)
df_crim["types"] = df_crim["types"].replace(19, "vloupání do bytu", regex=True)
df_crim["types"] = df_crim["types"].replace(20, "vloupání do rodinných domů", regex=True)
df_crim["types"] = df_crim["types"].replace(22, "vloupání do prodejny", regex=True)
df_crim["types"] = df_crim["types"].replace(23, "vloupání do restaurace", regex=True)
df_crim["types"] = df_crim["types"].replace(24, "vloupání do ubytovacích objektů", regex=True)
df_crim["types"] = df_crim["types"].replace(37, "krádeže motorových vozidel (dvoustopových)", regex=True)
df_crim["types"] = df_crim["types"].replace(38, "krádeže motorových vozidel (jednostopových)", regex=True)
df_crim["types"] = df_crim["types"].replace(41, "krádeže součástek aut", regex=True)
df_crim["types"] = df_crim["types"].replace(42, "krádeže jízdních kol", regex=True)
df_crim["types"] = df_crim["types"].replace(43, "krádeže na osobách", regex=True)
df_crim["types"] = df_crim["types"].replace(69, "nedovolené ozbrojování", regex=True)
df_crim["types"] = df_crim["types"].replace(71, "obecné ohrožení", regex=True)
df_crim["types"] = df_crim["types"].replace(79, "dopravní nehody", regex=True)
df_crim["types"] = df_crim["types"].replace(85, "střelná zbraň", regex=True)
df_crim["types"] = df_crim["types"].replace(86, "chladná zbraň", regex=True)
df_crim["types"] = df_crim["types"].replace(87, "výbušnina", regex=True)
df_crim["types"] = df_crim["types"].replace(88, "násilí proti skupině/jednotlivci", regex=True)


df_crim.to_csv('criminality.csv', sep=',', encoding='utf-8')
