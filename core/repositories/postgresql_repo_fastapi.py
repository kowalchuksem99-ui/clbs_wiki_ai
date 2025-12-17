# core/repositories/postgresql_repo_fastapi.py
from __future__ import annotations
import os, datetime as dt
from typing import Optional, List
from sqlalchemy import create_engine, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost/dbname")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    chats: Mapped[List["Chat"]] = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

class Chat(Base):
    __tablename__ = "chats"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[List["Message"]] = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(Integer, ForeignKey("chats.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)   # 'user' | 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")

# ---------- Repository ----------
class PostgreSQLRepository:
    """Репозиторий для FastAPI: методы принимают Session."""
    # --- Users ---
    @staticmethod
    def get_or_create_user(db: Session, username: str) -> User:
        u = db.query(User).filter(User.username == username).first()
        if not u:
            u = User(username=username)
            db.add(u); db.commit(); db.refresh(u)
        return u

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.get(User, int(user_id))

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    # --- Chats ---
    @staticmethod
    def create_chat(db: Session, user_id: int, title: str = "Новый чат") -> Chat:
        c = Chat(title=title, user_id=user_id)
        db.add(c); db.commit(); db.refresh(c)
        return c

    @staticmethod
    def get_user_chats(db: Session, user_id: int) -> List[Chat]:
        return (db.query(Chat)
                  .filter(Chat.user_id == user_id)
                  .order_by(Chat.updated_at.desc())
                  .all())

    @staticmethod
    def get_chat(db: Session, chat_id: int, user_id: int) -> Optional[Chat]:
        return (db.query(Chat)
                  .filter(Chat.id == chat_id, Chat.user_id == user_id)
                  .first())

    @staticmethod
    def update_chat_title(db: Session, chat: Chat, new_title: str) -> Chat:
        chat.title = new_title
        db.commit(); db.refresh(chat)
        return chat

    @staticmethod
    def delete_chat(db: Session, chat_id: int, user_id: int) -> bool:
        chat = PostgreSQLRepository.get_chat(db, chat_id, user_id)
        if chat:
            db.delete(chat); db.commit()
            return True
        return False

    # --- Messages ---
    @staticmethod
    def add_message(db: Session, chat_id: int, role: str, content: str) -> Message:
        m = Message(chat_id=chat_id, role=role, content=content)
        db.add(m)
        chat = db.get(Chat, chat_id)
        if chat:
            chat.updated_at = dt.datetime.utcnow()
        db.commit(); db.refresh(m)
        return m

    @staticmethod
    def get_chat_messages(db: Session, chat_id: int, user_id: int):
        chat = PostgreSQLRepository.get_chat(db, chat_id, user_id)
        if not chat:
            return None
        return (db.query(Message)
                  .filter(Message.chat_id == chat_id)
                  .order_by(Message.timestamp.asc())
                  .all())
