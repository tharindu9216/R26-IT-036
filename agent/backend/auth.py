"""Cookie authentication and role enforcement shared by every API route."""
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from accounts import AccountStore

COOKIE = 'sentivera_session'
DB_PATH = Path(os.environ.get('AGENT_ACCOUNTS_DB', Path(__file__).with_name('accounts.sqlite3')))
accounts = AccountStore(DB_PATH)
router = APIRouter(prefix='/api/auth')


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)
    role: Literal['patient', 'doctor'] = 'patient'


class Registration(BaseModel):
    email: str = Field(max_length=254)
    name: str = Field(max_length=100)
    password: str = Field(min_length=10, max_length=128)


class DoctorLink(BaseModel):
    email: str = Field(max_length=254)


def current_user(request: Request) -> dict:
    user = accounts.authenticate(request.cookies.get(COOKIE, ''))
    if not user:
        raise HTTPException(401, 'Please sign in to continue.')
    return user


def patient_id(request: Request) -> str:
    user = current_user(request)
    target = user['id'] if user['role'] == 'patient' else request.headers.get('X-Patient-ID', '')
    if not accounts.can_access(user, target):
        raise HTTPException(403, 'Select a patient who has shared access with you.')
    return target


def protect_api(request: Request) -> None:
    path = request.url.path
    # A custom header on writes forces cross-origin callers through CORS preflight.
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and request.headers.get('X-SentiVera-Client') != 'web':
        raise HTTPException(403, 'Missing request verification header.')
    if path in ('/api/auth/login', '/api/auth/register', '/api/health'):
        return
    user = current_user(request)
    doctor_only = path.startswith('/api/doctor') or path in ('/api/xai', '/api/deployment') or path.startswith('/api/sensor/')
    patient_only = path in ('/api/chat', '/api/chat/reset', '/api/voice/chat', '/api/questionnaire', '/api/auth/doctor')
    if doctor_only and user['role'] != 'doctor':
        raise HTTPException(403, 'This page is available to doctors only.')
    if patient_only and user['role'] != 'patient':
        raise HTTPException(403, 'This action is available to patients only.')


@router.post('/register')
def register(payload: Registration) -> dict:
    try:
        # A public registration can never provision a doctor account.
        return accounts.create(payload.email, payload.name, payload.password)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post('/login')
def login(payload: Credentials, response: Response) -> dict:
    try:
        token, user = accounts.login(payload.email, payload.password, payload.role)
    except ValueError as exc:
        raise HTTPException(401, str(exc)) from exc
    response.set_cookie(COOKIE, token, httponly=True, samesite='strict', max_age=43200,
                        secure=os.environ.get('AGENT_COOKIE_SECURE', '0') == '1', path='/')
    return user


@router.get('/me')
def me(request: Request) -> dict:
    return current_user(request)


@router.post('/logout')
def logout(request: Request, response: Response) -> dict:
    accounts.logout(request.cookies.get(COOKIE, ''))
    response.delete_cookie(COOKIE, path='/')
    return {'ok': True}


@router.post('/doctor')
def link_doctor(payload: DoctorLink, request: Request) -> dict:
    try:
        user = current_user(request)
        result = accounts.link_doctor(user['id'], payload.email)
        request.app.state.release_patient_sensor(user['id'])
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete('/doctor')
def unlink_doctor(request: Request) -> dict:
    user = current_user(request)
    accounts.unlink_doctor(user['id'])
    request.app.state.release_patient_sensor(user['id'])
    return {'ok': True}
