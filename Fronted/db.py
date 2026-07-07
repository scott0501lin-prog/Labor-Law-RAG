"""
MongoDB Atlas 資料層
負責：使用者帳號（註冊 / 登入驗證）、歷史對話紀錄的存取
"""

import os
import hashlib
from datetime import datetime, timezone

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_base_dir, ".env"))

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("MONGO_DB_NAME", "law_rag_db")

_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        if not MONGO_URI:
            raise RuntimeError("尚未設定 MONGO_URI，請在 .env 中加入 MongoDB Atlas 連線字串。")
        _client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    return _client


def get_db():
    return get_client()[DB_NAME]


def init_indexes():
    db = get_db()
    db.users.create_index("username", unique=True)
    db.chat_histories.create_index([("username", 1), ("updated_at", -1)])


# ==========================================
# 帳號模組
# ==========================================
def _hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    pw_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return pw_hash, salt


def register_user(username: str, password: str) -> tuple[bool, str]:
    pw_hash, salt = _hash_password(password)
    db = get_db()
    try:
        db.users.insert_one({
            "username": username,
            "password_hash": f"{salt}:{pw_hash}",
            "created_at": datetime.now(timezone.utc),
        })
        return True, "註冊成功！"
    except DuplicateKeyError:
        return False, "帳號已存在"


def verify_user(username: str, password: str) -> tuple[bool, str]:
    db = get_db()
    user = db.users.find_one({"username": username})
    if user is None:
        return False, "帳號或密碼錯誤"
    salt, stored_hash = user["password_hash"].split(":", 1)
    pw_hash, _ = _hash_password(password, salt)
    if pw_hash != stored_hash:
        return False, "帳號或密碼錯誤"
    return True, user["username"]


# ==========================================
# 歷史對話模組
# 一筆文件 = 一個對話串 (chat)，以 username + chat_id 為鍵
# ==========================================
def save_chat(username: str, chat_id: str, messages: list[dict]):
    db = get_db()
    db.chat_histories.update_one(
        {"username": username, "chat_id": chat_id},
        {
            "$set": {"messages": messages, "updated_at": datetime.now(timezone.utc)},
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )


def list_chats(username: str) -> list[dict]:
    """回傳該使用者所有對話的摘要（chat_id, updated_at），最新的在前。"""
    db = get_db()
    cursor = db.chat_histories.find(
        {"username": username},
        {"chat_id": 1, "updated_at": 1, "_id": 0},
    ).sort("updated_at", -1)
    return list(cursor)


def load_chat(username: str, chat_id: str) -> list[dict]:
    db = get_db()
    doc = db.chat_histories.find_one({"username": username, "chat_id": chat_id})
    return doc["messages"] if doc else []
