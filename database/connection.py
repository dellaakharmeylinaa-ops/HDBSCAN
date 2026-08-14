import mysql.connector

def get_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",      # isi password MySQL Anda
            database="agricluster"
        )

        if conn.is_connected():
            return conn

    except mysql.connector.Error as err:
        print("Koneksi gagal:", err)
        return None