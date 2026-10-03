import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

try:

    connection = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

    print("Database connected successfully!")

    cursor = connection.cursor()

    cursor.execute("SHOW TABLES")

    tables = cursor.fetchall()

    print("Tables:")

    for table in tables:
        print(table[0])

    cursor.close()
    connection.close()

except Exception as e:

    print("Database connection failed!")

    print(e)