import sqlite3
import json

# Data to be inserted
data = [
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583117456351246",
        "ceiling_name": "A口通道",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583117456351247",
        "ceiling_name": "B口通道",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583117456351252",
        "ceiling_name": "A口通道",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583117456351253",
        "ceiling_name": "B口通道",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583120174260248",
        "ceiling_name": "C口通道",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583120174260254",
        "ceiling_name": "C口通道",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583120178454549",
        "ceiling_name": "D口通道",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583120178454555",
        "ceiling_name": "D口通道",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "铝平板"
    },
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583120182648850",
        "ceiling_name": "站厅",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "铝方通"
    },
    {
        "line_id": "1552945383817203717",
        "line_name": "6号线",
        "station_id": "1554674973962960898",
        "station_name": "通州北关",
        "celling_id": "1939583120182648851",
        "ceiling_name": "站台",
        "depart_id": "1750773207980503042",
        "depart_name": "维护五中心",
        "ceiling_texture": "白色铝板"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583120182648856",
        "ceiling_name": "站厅",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "铝方通"
    },
    {
        "line_id": "1552945383800426498",
        "line_name": "昌平线",
        "station_id": "bfaadbb0b46c44e7b402789576e3b6e9",
        "station_name": "学院桥",
        "celling_id": "1939583120182648857",
        "ceiling_name": "站台",
        "depart_id": "1750773111364710401",
        "depart_name": "维护二中心",
        "ceiling_texture": "白色铝板"
    }
]

# Database path
db_path = 'D:/PythonCode/sql/example.db'

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if it does not exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS facilities (
    facility_id INTEGER PRIMARY KEY AUTOINCREMENT,
    load_log_id REAL,
    line_id TEXT,
    line_name TEXT,
    station_id TEXT,
    station_name TEXT,
    celling_id TEXT,
    ceiling_name TEXT,
    depart_id TEXT,
    depart_name TEXT,
    ceiling_texture TEXT
)
''')

# Insert data into the table
for entry in data:
    cursor.execute('''
    INSERT INTO facilities (line_id, line_name, station_id, station_name, celling_id, celling_name, depart_id, depart_name, ceiling_texture)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        entry['line_id'], entry['line_name'], entry['station_id'], entry['station_name'], entry['celling_id'], entry['ceiling_name'], entry['depart_id'], entry['depart_name'], entry['ceiling_texture']
    ))

# Commit and close connection
conn.commit()
conn.close()
