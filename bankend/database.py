import sqlite3
from pydantic import BaseModel

class Receipt(BaseModel):
    vendor: str
    date: str
    amount: float
    category: str = "Unknown"

def init_db():
    conn = sqlite3.connect("receipts.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY,
            vendor TEXT,
            date TEXT,
            amount REAL,
            category TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_receipt(receipt: Receipt):
    conn = sqlite3.connect("receipts.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO receipts (vendor, date, amount, category)
        VALUES (?, ?, ?, ?)
    """, (receipt.vendor, receipt.date, receipt.amount, receipt.category))
    conn.commit()
    conn.close()