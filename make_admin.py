import sqlite3
conn = sqlite3.connect('database/ppe_detection.db')
conn.execute("UPDATE users SET is_admin = 1, role = 'Admin' WHERE username = 'muhammad_ali'")
conn.commit()
print('Done - user is now Admin')
conn.close()
