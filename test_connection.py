from database.connection import get_connection

conn = get_connection()

if conn:
    print("Berhasil terhubung ke database!")
    conn.close()
else:
    print("Koneksi gagal.")