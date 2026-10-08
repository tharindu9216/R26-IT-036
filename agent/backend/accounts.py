"""Local accounts, opaque login sessions, and explicit doctor/patient links."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path


class AccountStore:
    def __init__(self, path: str | Path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
                password TEXT NOT NULL, doctor_id TEXT, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS logins (
                digest TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS login_attempts (
                email TEXT PRIMARY KEY, attempts INTEGER NOT NULL, since REAL NOT NULL
            );
        ''')
        self.db.commit()

    @staticmethod
    def password_hash(password: str, salt: str | None = None) -> str:
        salt = salt or secrets.token_hex(16)
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        return f'{salt}:{digest}'

    @staticmethod
    def public(row) -> dict:
        return {key: row[key] for key in ('id', 'email', 'name', 'role', 'doctor_id')}

    def create(self, email: str, name: str, password: str, role: str = 'patient') -> dict:
        email, name = email.strip().lower(), name.strip()
        if not name or len(name) > 100 or '@' not in email or len(email) > 254:
            raise ValueError('Enter a name and valid email address.')
        if not 10 <= len(password) <= 128:
            raise ValueError('Use a password between 10 and 128 characters.')
        if role not in ('patient', 'doctor'):
            raise ValueError('Invalid account role.')
        with self.lock:
            try:
                with self.db:
                    self.db.execute('INSERT INTO accounts VALUES (?, ?, ?, ?, ?, NULL, ?)',
                                    (uuid.uuid4().hex, email, name, role, self.password_hash(password), time.time()))
            except sqlite3.IntegrityError as exc:
                raise ValueError('An account already exists with that email.') from exc
            return self.public(self.db.execute('SELECT * FROM accounts WHERE email = ?', (email,)).fetchone())

    def login(self, email: str, password: str, role: str) -> tuple[str, dict]:
        email = email.strip().lower()
        with self.lock:
            now = time.time()
            attempt = self.db.execute('SELECT * FROM login_attempts WHERE email = ?', (email,)).fetchone()
            if attempt and attempt['since'] > now - 900 and attempt['attempts'] >= 10:
                raise ValueError('Too many attempts. Please try again in 15 minutes.')
            row = self.db.execute('SELECT * FROM accounts WHERE email = ?', (email,)).fetchone()
            stored = row['password'] if row else '00' * 16 + ':' + '00' * 64
            valid = hmac.compare_digest(self.password_hash(password, stored.split(':')[0]), stored)
            if not row or not valid or row['role'] != role:
                with self.db:
                    self.db.execute('''INSERT INTO login_attempts VALUES (?, 1, ?)
                        ON CONFLICT(email) DO UPDATE SET
                        attempts = CASE WHEN since < ? THEN 1 ELSE attempts + 1 END,
                        since = CASE WHEN since < ? THEN excluded.since ELSE since END''',
                        (email, now, now - 900, now - 900))
                raise ValueError('Email or password is incorrect for this portal.')
            token = secrets.token_urlsafe(32)
            with self.db:
                self.db.execute('DELETE FROM login_attempts WHERE email = ?', (email,))
                self.db.execute('DELETE FROM logins WHERE expires_at < ?', (now,))
                self.db.execute('INSERT INTO logins VALUES (?, ?, ?)',
                                (hashlib.sha256(token.encode()).hexdigest(), row['id'], now + 43200))
            return token, self.public(row)

    def authenticate(self, token: str) -> dict | None:
        with self.lock:
            row = self.db.execute('''SELECT accounts.* FROM accounts JOIN logins
                ON accounts.id = logins.account_id WHERE digest = ? AND expires_at > ?''',
                (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
            return self.public(row) if row else None

    def logout(self, token: str) -> None:
        with self.lock, self.db:
            self.db.execute('DELETE FROM logins WHERE digest = ?', (hashlib.sha256(token.encode()).hexdigest(),))

    def link_doctor(self, patient_id: str, email: str) -> dict:
        with self.lock, self.db:
            doctor = self.db.execute("SELECT * FROM accounts WHERE email = ? AND role = 'doctor'",
                                     (email.strip().lower(),)).fetchone()
            if not doctor:
                raise ValueError('No doctor account found with that email.')
            self.db.execute("UPDATE accounts SET doctor_id = ? WHERE id = ? AND role = 'patient'",
                            (doctor['id'], patient_id))
            return {'name': doctor['name'], 'email': doctor['email']}

    def unlink_doctor(self, patient_id: str) -> None:
        with self.lock, self.db:
            self.db.execute('UPDATE accounts SET doctor_id = NULL WHERE id = ?', (patient_id,))

    def patients(self, doctor_id: str) -> list[dict]:
        with self.lock:
            return [self.public(row) for row in self.db.execute(
                "SELECT * FROM accounts WHERE doctor_id = ? AND role = 'patient' ORDER BY name", (doctor_id,))]

    def can_access(self, user: dict, patient_id: str) -> bool:
        if user['role'] == 'patient':
            return user['id'] == patient_id
        with self.lock:
            return self.db.execute("SELECT 1 FROM accounts WHERE id = ? AND doctor_id = ? AND role = 'patient'",
                                   (patient_id, user['id'])).fetchone() is not None
