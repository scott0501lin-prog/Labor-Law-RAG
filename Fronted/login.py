import sqlite3
import hashlib
import os
from flask import Flask, request, jsonify, session

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "message": "帳號和密碼不能為空"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "message": "密碼至少需要 6 個字元"}), 400

    pw_hash, salt = hash_password(password)
    stored = f"{salt}:{pw_hash}"

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, stored),
            )
        return jsonify({"success": True, "message": "註冊成功"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "帳號已存在"}), 409


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"success": False, "message": "帳號和密碼不能為空"}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        return jsonify({"success": False, "message": "帳號或密碼錯誤"}), 401

    salt, stored_hash = row["password_hash"].split(":", 1)
    pw_hash, _ = hash_password(password, salt)

    if pw_hash != stored_hash:
        return jsonify({"success": False, "message": "帳號或密碼錯誤"}), 401

    session["user_id"] = row["id"]
    session["username"] = row["username"]
    return jsonify({"success": True, "message": f"歡迎回來，{username}！"}), 200


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "已登出"}), 200


@app.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"success": False, "message": "尚未登入"}), 401
    return jsonify({"success": True, "username": session["username"]}), 200


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
