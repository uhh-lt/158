import psycopg2

# Подключение к базе данных
con = psycopg2.connect(
  user="158_user",
  password="158",
  database="fasttext_vectors",
  host="127.0.0.1",
  port="5433"
)

print("Database opened successfully")

# Создание таблицы
cur = con.cursor()
cur.execute("SELECT * from en_ WHERE word = 'the'")

rows = cur.fetchall()


print("Table created successfully")
#con.commit()
con.close()
