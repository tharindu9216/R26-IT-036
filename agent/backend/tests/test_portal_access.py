"""Portal isolation tests; inference is stubbed, no model weights are loaded."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import api
import auth
from accounts import AccountStore
from chat_store import ChatStore


class PortalAccessTests(unittest.TestCase):
    def setUp(self):
        self.accounts = AccountStore(':memory:')
        self.store = ChatStore(':memory:')
        self.addCleanup(self.accounts.db.close)
        self.addCleanup(self.store.close)
        for target, name, value in (
            (auth, 'accounts', self.accounts), (api, 'accounts', self.accounts),
            (api, '_store', self.store), (api, '_turn_contexts', {}),
            (api, '_voice_audio_outputs', {}), (api, '_questionnaire_results', {}),
            (api, '_c1_service', None), (api, '_sensor_patient_id', None), (api, '_sensor_skipped', False),
        ):
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.patient = self.accounts.create('patient@example.com', 'Patient One', 'patient-pass-123')
        self.other = self.accounts.create('other@example.com', 'Patient Two', 'patient-pass-123')
        self.doctor = self.accounts.create('doctor@example.com', 'Doctor One', 'doctor-pass-123', 'doctor')
        self.other_doctor = self.accounts.create('otherdoc@example.com', 'Doctor Two', 'doctor-pass-123', 'doctor')
        self.client = TestClient(api.app, headers={'X-SentiVera-Client': 'web'})
        self.addCleanup(self.client.close)

    def login(self, account=None):
        account = account or self.patient
        response = self.client.post('/api/auth/login', json={
            'email': account['email'], 'password': f"{account['role']}-pass-123", 'role': account['role'],
        })
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def link(self):
        self.accounts.link_doctor(self.patient['id'], self.doctor['email'])
        return {'X-Patient-ID': self.patient['id']}

    def test_anonymous_endpoints_are_protected(self):
        for path in ('/api/chat/history', '/api/doctor/patients', '/api/sensor/status', '/api/models/text/status'):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.get('/api/health').status_code, 200)

    def test_registration_cannot_create_doctor(self):
        response = self.client.post('/api/auth/register', json={
            'email': 'new@example.com', 'name': 'New User', 'password': 'new-password-123', 'role': 'doctor',
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['role'], 'patient')
        self.assertNotIn('password', response.json())

    def test_cookie_login_wrong_portal_logout_and_revocation(self):
        response = self.login()
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('SameSite=strict', response.headers['set-cookie'])
        token = self.client.cookies.get(auth.COOKIE)
        self.assertEqual(self.client.get('/api/auth/me').json()['id'], self.patient['id'])
        wrong = self.client.post('/api/auth/login', json={
            'email': self.patient['email'], 'password': 'patient-pass-123', 'role': 'doctor',
        })
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(self.client.post('/api/auth/logout').status_code, 200)
        self.assertIsNone(self.accounts.authenticate(token))
        self.assertEqual(self.client.get('/api/auth/me').status_code, 401)

    def test_csrf_header_is_required(self):
        with TestClient(api.app) as client:
            self.assertEqual(client.post('/api/auth/login', json={
                'email': self.patient['email'], 'password': 'patient-pass-123',
            }).status_code, 403)

    def test_patient_cannot_read_doctor_endpoints(self):
        self.login()
        for path in ('/api/doctor/patients', '/api/doctor/activity', '/api/sensor/status', '/api/deployment'):
            self.assertEqual(self.client.get(path).status_code, 403)
        self.assertEqual(self.client.post('/api/xai', json={'message': 'hello', 'turn_id': 'turn'}).status_code, 403)
        self.assertEqual(self.client.post('/api/sensor/start', json={}).status_code, 403)

    def test_session_id_and_header_cannot_impersonate_other_patient(self):
        self.store.append(self.patient['id'], 'user', 'my message')
        self.store.append(self.other['id'], 'user', 'private other message')
        self.login()
        response = self.client.get('/api/chat/history', params={'session_id': self.other['id']},
                                   headers={'X-Patient-ID': self.other['id']})
        self.assertEqual([m['text'] for m in response.json()], ['my message'])
        self.client.post('/api/chat/reset', json={'session_id': self.other['id']})
        self.assertEqual(self.store.history(self.patient['id']), [])
        self.assertEqual(self.store.history(self.other['id']), [('user', 'private other message')])

    def test_doctors_only_see_linked_patients_and_access_is_revocable(self):
        headers = self.link()
        self.store.append(self.patient['id'], 'assistant', 'reply', trace={'steps': [], 'route': 'base'})
        self.login(self.doctor)
        patients = self.client.get('/api/doctor/patients').json()
        self.assertEqual([p['id'] for p in patients], [self.patient['id']])
        self.assertEqual(patients[0]['message_count'], 1)
        self.assertEqual(self.client.get('/api/doctor/activity', headers=headers).status_code, 200)
        self.assertEqual(self.client.get('/api/doctor/activity', headers={'X-Patient-ID': self.other['id']}).status_code, 403)
        self.accounts.unlink_doctor(self.patient['id'])
        self.assertEqual(self.client.get('/api/doctor/activity', headers=headers).status_code, 403)
        self.assertEqual(self.client.get('/api/doctor/patients').json(), [])

    def test_doctor_cannot_chat_or_modify_patient_checkin(self):
        self.login(self.doctor)
        self.assertEqual(self.client.post('/api/chat', json={'message': 'hello'}).status_code, 403)
        self.assertEqual(self.client.post('/api/chat/reset', json={}).status_code, 403)
        self.assertEqual(self.client.post('/api/questionnaire', json={'answers': {}}).status_code, 403)

    def test_text_reply_redacts_analysis_but_doctor_activity_keeps_trace(self):
        self.link()
        self.login()
        graph = Mock()
        graph.invoke.return_value = {'reply': 'I hear you.', 'current_emotion': 'sadness', 'reply_route': 'supportive'}
        trace = {'version': 2, 'mode': 'text', 'route': 'supportive', 'steps': []}
        with patch.object(api, 'get_workflow', return_value=graph), patch.object(api, 'build_trace', return_value=trace):
            result = self.client.post('/api/chat', json={'message': 'I feel tired', 'session_id': self.other['id']})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(set(result.json()), {'turn_id', 'reply', 'support_contacts'})
        self.assertEqual(self.store.history(self.other['id']), [])
        self.assertEqual(len(self.store.history(self.patient['id'])), 2)
        self.assertFalse(self.client.get('/api/chat/history').json()[-1].get('trace'))
        self.login(self.doctor)
        activity = self.client.get('/api/doctor/activity', headers={'X-Patient-ID': self.patient['id']}).json()
        self.assertEqual(activity[-1]['trace']['route'], 'supportive')
        self.assertTrue(activity[-1]['explanation_available'])

    def test_voice_reply_uses_account_and_redacts_analysis(self):
        self.login()
        appraisal = Mock(voice_stress_score=0.1, voice_stress=False, dominant_state='neutral', uncertain=False)
        appraisal.as_dict.return_value = {'dominant_state': 'neutral'}
        analyzer, transcriber, synthesizer, graph = Mock(), Mock(), Mock(), Mock()
        analyzer.predict.return_value = appraisal
        transcriber.transcribe.return_value = SimpleNamespace(text='Hello there', language='en', language_probability=1.0, duration_seconds=1.0)
        synthesizer.synthesize.return_value = b'fake-wave'
        graph.invoke.return_value = {'reply': 'Hello, I am here.', 'current_emotion': 'neutral'}
        c2_stub = SimpleNamespace(decode_audio=lambda _: SimpleNamespace(size=16000))
        with patch.dict('sys.modules', {'c2': c2_stub}), patch.object(api, 'get_voice_services', return_value=(analyzer, transcriber, synthesizer)), patch.object(api, 'get_voice_workflow', return_value=graph):
            response = self.client.post('/api/voice/chat', files={'audio': ('input.wav', b'input', 'audio/wav')}, data={'session_id': self.other['id']})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(set(result), {'turn_id', 'reply', 'transcript', 'reply_audio_url', 'duration_seconds', 'support_contacts'})
        self.assertEqual(result['transcript'], 'Hello there')
        self.assertEqual(self.client.get(result['reply_audio_url']).content, b'fake-wave')
        self.assertEqual(len(self.store.history(self.patient['id'])), 2)
        self.assertEqual(self.store.history(self.other['id']), [])
        self.assertEqual(self.store.transcript(self.patient['id'])[-1]['trace']['mode'], 'voice')

    def test_turn_audio_and_xai_require_patient_access(self):
        api._turn_contexts['turn'] = {'patient_id': self.patient['id'], 'message': 'hello'}
        api._voice_audio_outputs['turn'] = b'fake-wave'
        self.login(self.other)
        self.assertEqual(self.client.get('/api/voice/audio/turn').status_code, 404)
        self.login(self.patient)
        self.assertEqual(self.client.get('/api/voice/audio/turn').content, b'fake-wave')
        self.login(self.other_doctor)
        self.assertEqual(self.client.post('/api/xai', json={'message': 'hello', 'turn_id': 'turn'}).status_code, 404)
        self.login(self.doctor)
        self.assertEqual(self.client.post('/api/xai', json={'message': 'hello'}).status_code, 400)

    def test_reset_revokes_saved_audio_and_explanation(self):
        api._turn_contexts['turn'] = {'patient_id': self.patient['id'], 'message': 'hello'}
        api._voice_audio_outputs['turn'] = b'fake-wave'
        self.login()
        self.client.post('/api/chat/reset', json={})
        self.assertEqual(self.client.get('/api/voice/audio/turn').status_code, 404)

    def test_shared_sensor_is_not_used_for_another_patient(self):
        api._sensor_patient_id = self.patient['id']
        api._c1_service = Mock()
        state, context = api._sensor_turn_data(self.other['id'])
        self.assertFalse(state['sensor_stress_available'])
        api._c1_service.snapshot.assert_not_called()
        self.assertEqual(context['sensor_predictions'], ())

    def test_doctor_cannot_take_or_read_another_patient_sensor(self):
        self.link()
        api._sensor_patient_id = self.other['id']
        api._c1_service = Mock()
        self.login(self.doctor)
        headers = {'X-Patient-ID': self.patient['id']}
        self.assertEqual(self.client.get('/api/sensor/status', headers=headers).json(), {'mode': 'idle', 'ready': False})
        self.assertEqual(self.client.post('/api/sensor/disconnect', headers=headers).status_code, 409)
        api._c1_service.stop.assert_not_called()

    def test_patient_can_link_and_revoke_doctor(self):
        self.login()
        self.assertEqual(self.client.post('/api/auth/doctor', json={'email': self.doctor['email']}).status_code, 200)
        self.assertEqual(self.client.get('/api/auth/me').json()['doctor_id'], self.doctor['id'])
        self.assertEqual(self.client.delete('/api/auth/doctor').status_code, 200)
        self.assertIsNone(self.client.get('/api/auth/me').json()['doctor_id'])

    def test_revoking_access_stops_patient_sensor(self):
        self.link()
        self.login()
        sensor = Mock()
        api._sensor_patient_id = self.patient['id']
        api._c1_service = sensor
        self.assertEqual(self.client.delete('/api/auth/doctor').status_code, 200)
        sensor.stop.assert_called_once()
        self.assertIsNone(api._c1_service)
        self.assertIsNone(api._sensor_patient_id)

    def test_password_hashes_and_session_expiry(self):
        stored = self.accounts.db.execute('SELECT password FROM accounts WHERE id = ?', (self.patient['id'],)).fetchone()[0]
        self.assertNotIn('patient-pass-123', stored)
        token, _ = self.accounts.login(self.patient['email'], 'patient-pass-123', 'patient')
        self.accounts.db.execute('UPDATE logins SET expires_at = 0')
        self.accounts.db.commit()
        self.assertIsNone(self.accounts.authenticate(token))


if __name__ == '__main__':
    unittest.main()
