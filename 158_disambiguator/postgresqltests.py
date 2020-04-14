import psycopg2

# Подключение к базе данных
con = psycopg2.connect(
    database="postgres",
    user="postgres",
    password="",
    host="127.0.0.1",
    port="5432"
)

print("Database opened successfully")

# Создание таблицы
cur = con.cursor()
cur.execute('''CREATE TABLE STUDENT  
     (ADMISSION INT PRIMARY KEY NOT NULL,
     NAME TEXT NOT NULL,
     AGE INT NOT NULL,
     COURSE CHAR(50),
     DEPARTMENT CHAR(50));''')

print("Table created successfully")
con.commit()
con.close()

# Вставка данных
cur = con.cursor()
cur.execute(
    "INSERT INTO STUDENT (ADMISSION,NAME,AGE,COURSE,DEPARTMENT) VALUES (3420, 'John', 18, 'Computer Science', 'ICT')"
)

con.commit()

# Извлечение данных
cur.execute("SELECT admission, name, age, course, department from STUDENT")

rows = cur.fetchall()
for row in rows:
    print("ADMISSION =", row[0])
    print("NAME =", row[1])
    print("AGE =", row[2])
    print("COURSE =", row[3])
    print("DEPARTMENT =", row[4], "\n")

print("Operation done successfully")
con.close()

# Обновление таблиц
cur = con.cursor()
cur.execute("UPDATE STUDENT set AGE = 20 where ADMISSION = 3420")
con.commit()

# Удаление строк
cur = con.cursor()
cur.execute("DELETE from STUDENT where ADMISSION=3420;")
con.commit()

print("Total deleted rows:", cur.rowcount)

# Pandas df to PostgreSQL
from sqlalchemy import create_engine

engine = create_engine('postgresql://scott:tiger@localhost:5432/mydatabase')
df.to_sql('table_name', engine)
