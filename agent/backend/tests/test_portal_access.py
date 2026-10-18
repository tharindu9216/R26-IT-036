"""Portal isolation tests; inference is stubbed, no model weights are loaded."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
import api
import auth
import settings
import dass21
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
            (settings, 'API_PRELOAD_MODELS', False),
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

    def ready_sensor(self, *, assessment=True):
        from c1 import ServiceState

        snapshot = SimpleNamespace(
            state=ServiceState.RUNNING, stream_stalled=False, last_error=None,
            calibration=SimpleNamespace(), last_prediction=SimpleNamespace(),
            recent_predictions=(), smoothed_is_stressed=False, smoothed_probability=0.2,
            calibration_remaining_seconds=0.0, warmup_remaining_seconds=0.0,
        )
        api._sensor_patient_id = self.patient['id']
        api._c1_service = SimpleNamespace(snapshot=lambda: snapshot, predictor=SimpleNamespace())
        if assessment and not self.store.dass21_history(self.patient['id']):
            answers = dict.fromkeys((str(i) for i in range(1, 22)), 0)
            self.store.save_dass21(self.patient['id'], dass21.VERSION, answers, dass21.score_answers(answers))
        return snapshot

    def test_doctor_directory_requires_patient_login_and_exposes_contacts_only(self):
        self.assertEqual(self.client.get('/api/auth/doctors').status_code, 401)
        self.login()
        response = self.client.get('/api/auth/doctors')
        self.assertEqual(response.status_code, 200)
        doctors = response.json()
        self.assertEqual([item['id'] for item in doctors], [self.doctor['id'], self.other_doctor['id']])
        for item in doctors:
            self.assertEqual(set(item), {'id', 'name', 'email'})
        self.assertIsNone(self.client.get('/api/auth/me').json()['doctor_id'])
        self.login(self.doctor)
        self.assertEqual(self.client.get('/api/auth/doctors').status_code, 403)

    def test_directory_selection_links_only_after_confirmation_and_revokes_access(self):
        self.login()
        directory = self.client.get('/api/auth/doctors').json()
        selected = next(item for item in directory if item['id'] == self.doctor['id'])
        self.assertEqual(self.accounts.patients(self.doctor['id']), [])
        response = self.client.post('/api/auth/doctor', json={'email': selected['email']})
        self.assertEqual(response.status_code, 200, response.text)
        self.login(self.doctor)
        patients = self.client.get('/api/doctor/patients').json()
        self.assertEqual([item['id'] for item in patients], [self.patient['id']])
        self.login(self.other_doctor)
        self.assertEqual(self.client.get('/api/doctor/patients').json(), [])
        self.login()
        self.assertEqual(self.client.delete('/api/auth/doctor').status_code, 200)
        self.login(self.doctor)
        self.assertEqual(self.client.get('/api/doctor/patients').json(), [])

    def test_directory_includes_newly_provisioned_doctors_on_refresh(self):
        self.login()
        self.assertEqual(len(self.client.get('/api/auth/doctors').json()), 2)
        doctor = self.accounts.create('new@example.com', 'A New Doctor', 'doctor-pass-123', 'doctor')
        doctors = self.client.get('/api/auth/doctors').json()
        self.assertEqual(len(doctors), 3)
        self.assertEqual(doctors[0], {key: doctor[key] for key in ('id', 'name', 'email')})

    def test_dass21_patient_completion_hides_results_and_doctor_can_review(self):
        self.login()
        form = self.client.get('/api/assessments/dass21/form').json()
        self.assertEqual(len(form['questions']), 21)
        self.assertEqual([option['value'] for option in form['options']], [0, 1, 2, 3])
        self.assertFalse(self.client.get('/api/assessments/dass21/status').json()['completed'])
        answers = dict.fromkeys((str(i) for i in range(1, 22)), 2)
        response = self.client.post('/api/assessments/dass21', json={'answers': answers, 'session_id': self.other['id']})
        self.assertEqual(response.status_code, 200, response.text)
        allowed = {'completed', 'assessment_id', 'completed_at', 'valid_until'}
        self.assertEqual(set(response.json()), allowed)
        self.assertEqual(set(self.client.get('/api/assessments/dass21/status').json()), allowed)
        self.assertEqual(self.store.dass21_history(self.other['id']), [])
        self.assertEqual(self.client.get('/api/doctor/assessments/dass21').status_code, 403)
        headers = self.link()
        self.login(self.doctor)
        history = self.client.get('/api/doctor/assessments/dass21', headers=headers).json()['assessments']
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['answers'], answers)
        self.assertEqual(history[0]['scores']['stress'], {'raw_score': 14, 'score': 28, 'severity': 'Severe'})
        self.assertEqual(self.client.post('/api/assessments/dass21', json={'answers': answers}).status_code, 403)
        self.assertEqual(self.client.get('/api/doctor/assessments/dass21', headers={'X-Patient-ID': self.other['id']}).status_code, 403)
        self.accounts.unlink_doctor(self.patient['id'])
        self.assertEqual(self.client.get('/api/doctor/assessments/dass21', headers=headers).status_code, 403)

    def test_dass21_invalid_submission_is_not_saved(self):
        self.login()
        good = dict.fromkeys((str(i) for i in range(1, 22)), 0)
        for answers in ({}, {**good, '22': 1}, {**good, '1': 4}, {**good, '1': True}, {**good, '1': '2'}, {**good, '1': 1.5}):
            response = self.client.post('/api/assessments/dass21', json={'answers': answers})
            self.assertIn(response.status_code, (400, 422), response.text)
        self.assertEqual(self.store.dass21_history(self.patient['id']), [])

    def test_both_modes_require_completed_current_dass21(self):
        self.login()
        self.ready_sensor(assessment=False)
        self.assertEqual(self.client.get('/api/chat/readiness').json()['state'], 'waiting_for_assessment')
        self.assert_chat_blocked()
        answers = dict.fromkeys((str(i) for i in range(1, 22)), 0)
        self.assertEqual(self.client.post('/api/assessments/dass21', json={'answers': answers}).status_code, 200)
        self.assertTrue(self.client.get('/api/chat/readiness').json()['ready'])
        with patch.object(dass21, 'REASSESS_AFTER_SECONDS', 0):
            self.assertFalse(self.client.get('/api/assessments/dass21/status').json()['completed'])
            self.assertFalse(self.client.get('/api/chat/readiness').json()['ready'])
            self.assert_chat_blocked()

    def test_dass21_and_legacy_checkin_never_enter_live_stress_fusion(self):
        self.login()
        answers = dict.fromkeys((str(i) for i in range(1, 22)), 3)
        self.client.post('/api/assessments/dass21', json={'answers': answers})
        legacy = dict(manageable=0, tense=3, overwhelmed=3, relaxed=0, worried=3, in_control=0)
        self.client.post('/api/questionnaire', json={'answers': legacy})
        state, context = api._multimodal_turn_data(self.patient['id'], {
            'sensor_stress_available': True, 'sensor_stress_probability': 0.2,
        }, voice_probability=0.9)
        self.assertEqual(state['multimodal_stress_probability'], 0.2)
        self.assertEqual(state['multimodal_stress_weights'], {'sensor': 1.0})
        self.assertFalse(state['questionnaire_stress_available'])
        self.assertIsNone(context['questionnaire'])

    def assert_chat_blocked(self):
        with patch.object(api, 'get_workflow') as text_models, patch.object(api, 'get_voice_services') as voice_models:
            responses = [
                self.client.post('/api/chat', json={'message': 'hello'}),
                self.client.post('/api/chat/face', data={'message': 'hello'}),
                self.client.post('/api/voice/chat', files={'audio': ('input.wav', b'input', 'audio/wav')}),
            ]
            for response in responses:
                self.assertEqual(response.status_code, 409, response.text)
            text_models.assert_not_called()
            voice_models.assert_not_called()

    def test_chat_requires_sensor_even_after_questionnaire(self):
        self.login()
        answers = dict.fromkeys(('manageable', 'tense', 'overwhelmed', 'relaxed', 'worried', 'in_control'), 1)
        self.assertEqual(self.client.post('/api/questionnaire', json={'answers': answers}).status_code, 200)
        self.assertFalse(self.client.get('/api/chat/readiness').json()['ready'])
        self.assert_chat_blocked()

    def test_both_paths_wait_for_calibration_and_first_prediction(self):
        from c1 import ServiceState

        self.login()
        snapshot = self.ready_sensor()
        for state, calibration, prediction in (
            (ServiceState.CALIBRATING, None, None),
            (ServiceState.WARMING_UP, SimpleNamespace(), None),
            (ServiceState.RUNNING, SimpleNamespace(), None),
        ):
            with self.subTest(state=state):
                snapshot.state, snapshot.calibration, snapshot.last_prediction = state, calibration, prediction
                snapshot.calibration_remaining_seconds = 120.0 if state == ServiceState.CALIBRATING else 0.0
                snapshot.warmup_remaining_seconds = 30.0 if state == ServiceState.WARMING_UP else 0.0
                status = self.client.get('/api/chat/readiness').json()
                self.assertFalse(status['ready'])
                self.assertEqual(status['calibration_remaining_seconds'], snapshot.calibration_remaining_seconds)
                self.assertEqual(status['warmup_remaining_seconds'], snapshot.warmup_remaining_seconds)
                self.assert_chat_blocked()

    def test_sensor_readiness_is_patient_scoped_and_cannot_be_skipped(self):
        self.login()
        self.ready_sensor()
        api._sensor_patient_id = self.other['id']
        response = self.client.get('/api/chat/readiness', params={'session_id': self.other['id']},
                                   headers={'X-Patient-ID': self.other['id']})
        self.assertEqual(response.json()['state'], 'waiting_for_sensor')
        self.assert_chat_blocked()
        api._sensor_patient_id = self.patient['id']
        api._sensor_skipped = True
        self.assertFalse(self.client.get('/api/chat/readiness').json()['ready'])
        self.assert_chat_blocked()

    def test_ready_sensor_status_exposes_no_predictions_and_rechecks_failure(self):
        from c1 import ServiceState

        self.login()
        snapshot = self.ready_sensor()
        ready = self.client.get('/api/chat/readiness').json()
        self.assertTrue(ready['ready'])
        self.assertNotIn('sensor_stress_probability', ready)
        self.assertNotIn('sensor_waveform', ready)
        for changes in ({'stream_stalled': True}, {'last_error': 'bad signal'}, {'state': ServiceState.STOPPED}):
            snapshot = self.ready_sensor()
            for key, value in changes.items():
                setattr(snapshot, key, value)
            self.assertFalse(self.client.get('/api/chat/readiness').json()['ready'])
            self.assert_chat_blocked()

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

    def test_doctor_summary_covers_linked_patients_only(self):
        headers = self.link()
        self.login()
        self.ready_sensor()
        graph = Mock()
        graph.invoke.return_value = {
            'reply': 'I hear you.', 'current_emotion': 'sadness', 'text_signal_probability': 0.7,
            'multimodal_stress_available': True, 'multimodal_stress_probability': 0.6,
        }
        with patch.object(api, 'get_workflow', return_value=graph):
            self.assertEqual(self.client.post('/api/chat', json={'message': 'rough day'}).status_code, 200)
            # Patients can never post a camera turn as a doctor, or vice versa.
            self.assertEqual(self.client.get('/api/doctor/summary').status_code, 403)
        self.client.post('/api/chat/reset', json={})
        self.login(self.doctor)
        self.assertEqual(self.client.post('/api/chat/face', data={'message': 'hi'}, headers=headers).status_code, 403)
        summary = self.client.get('/api/doctor/summary?days=7', headers=headers)
        self.assertEqual(summary.status_code, 200, summary.text)
        body = summary.json()
        # New chat cleared the conversation but not the wellbeing history.
        self.assertEqual(body['totals']['check_ins'], 1)
        self.assertEqual(body['emotions'][0]['label'], 'sadness')
        self.assertEqual({s['key']: s['average'] for s in body['sources']}['text'], 0.7)
        self.assertEqual(self.client.get('/api/doctor/summary', headers={'X-Patient-ID': self.other['id']}).status_code, 403)
        self.assertEqual(self.client.get('/api/doctor/summary?days=400', headers=headers).status_code, 400)

    def test_text_reply_redacts_analysis_but_doctor_activity_keeps_trace(self):
        self.link()
        self.login()
        self.ready_sensor()
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
        # A backend restart empties the in-memory cache; the stored context
        # keeps the explanation available.
        api._turn_contexts.clear()
        activity = self.client.get('/api/doctor/activity', headers={'X-Patient-ID': self.patient['id']}).json()
        self.assertTrue(activity[-1]['explanation_available'])
        restored = api._turn_context(result.json()['turn_id'])
        self.assertEqual((restored['message'], restored['patient_id']), ('I feel tired', self.patient['id']))

    def test_voice_reply_uses_account_and_redacts_analysis(self):
        self.login()
        self.ready_sensor()
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

    def test_voice_rechecks_sensor_if_recalibrated_during_transcription(self):
        from c1 import ServiceState

        self.login()
        snapshot = self.ready_sensor()
        analyzer, transcriber, synthesizer, graph = Mock(), Mock(), Mock(), Mock()

        def transcribe(_waveform):
            snapshot.state = ServiceState.CALIBRATING
            snapshot.calibration = None
            return SimpleNamespace(text='Hello there')

        transcriber.transcribe.side_effect = transcribe
        c2_stub = SimpleNamespace(decode_audio=lambda _: SimpleNamespace(size=16000))
        with patch.dict('sys.modules', {'c2': c2_stub}), patch.object(api, 'get_voice_services', return_value=(analyzer, transcriber, synthesizer)), patch.object(api, 'get_voice_workflow', return_value=graph):
            response = self.client.post('/api/voice/chat', files={'audio': ('input.wav', b'input', 'audio/wav')})
        self.assertEqual(response.status_code, 409, response.text)
        graph.invoke.assert_not_called()
        synthesizer.synthesize.assert_not_called()

    def test_voice_turn_scores_camera_frames_without_exposing_them(self):
        from facial import aggregate_frame_probabilities

        self.login()
        self.ready_sensor()
        appraisal = Mock(voice_stress_score=0.1, voice_stress=False, dominant_state='neutral', uncertain=False)
        appraisal.as_dict.return_value = {'dominant_state': 'neutral'}
        analyzer, transcriber, synthesizer, graph = Mock(), Mock(), Mock(), Mock()
        analyzer.predict.return_value = appraisal
        transcriber.transcribe.return_value = SimpleNamespace(text='Hello there', language='en', language_probability=1.0, duration_seconds=1.0)
        synthesizer.synthesize.return_value = b'fake-wave'
        graph.invoke.return_value = {'reply': 'Hello, I am here.', 'current_emotion': 'neutral'}
        face = Mock()
        face.predict_frames.return_value = aggregate_frame_probabilities(
            [0.7, 0.8, 0.9], frames_received=4, min_face_frames=3, threshold=0.5)
        c2_stub = SimpleNamespace(decode_audio=lambda _: SimpleNamespace(size=16000))
        frames = [('face_frames', (f'f{i}.jpg', b'jpeg', 'image/jpeg')) for i in range(4)]
        with patch.dict('sys.modules', {'c2': c2_stub}), patch.object(api, 'get_voice_services', return_value=(analyzer, transcriber, synthesizer)), patch.object(api, 'get_voice_workflow', return_value=graph), patch.object(api, 'get_facial_analyzer', return_value=face):
            response = self.client.post('/api/voice/chat', files=[('audio', ('input.wav', b'input', 'audio/wav')), *frames])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(face.predict_frames.call_args.args[0], [b'jpeg'] * 4)
        state = graph.invoke.call_args.args[0]
        self.assertTrue(state['face_stress_available'])
        self.assertAlmostEqual(state['face_stress_probability'], 0.8)
        self.assertEqual(state['face_frames_detected'], 3)
        self.assertNotIn('face_stress_probability', response.json())

    def test_typed_turn_scores_camera_frames_without_exposing_them(self):
        from facial import aggregate_frame_probabilities

        self.login()
        self.ready_sensor()
        graph = Mock()
        graph.invoke.return_value = {'reply': 'I am here with you.', 'current_emotion': 'sadness'}
        face = Mock()
        face.predict_frames.return_value = aggregate_frame_probabilities(
            [0.7, 0.8, 0.9], frames_received=4, min_face_frames=3, threshold=0.5)
        frames = [('face_frames', (f'f{i}.jpg', b'jpeg', 'image/jpeg')) for i in range(4)]
        with patch.object(api, 'get_workflow', return_value=graph), patch.object(api, 'get_facial_analyzer', return_value=face):
            response = self.client.post('/api/chat/face', data={'message': 'I feel low'}, files=frames)
            plain = self.client.post('/api/chat', json={'message': 'still low'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(face.predict_frames.call_args.args[0], [b'jpeg'] * 4)
        state = graph.invoke.call_args_list[0].args[0]
        self.assertEqual(state['user_text'], 'I feel low')
        self.assertTrue(state['face_stress_available'])
        self.assertAlmostEqual(state['face_stress_probability'], 0.8)
        self.assertNotIn('face_stress_probability', response.json())
        # The JSON route still works and simply has no camera reading.
        self.assertEqual(plain.status_code, 200, plain.text)
        self.assertFalse(graph.invoke.call_args_list[1].args[0]['face_stress_available'])
        trace = self.store.transcript(self.patient['id'])[1]['trace']
        self.assertEqual(trace['mode'], 'text')
        self.assertIn('face', [step['key'] for step in trace['steps']])

    def test_voice_turn_without_camera_is_no_reading(self):
        state, context = api._facial_turn_data([])
        self.assertFalse(state['face_stress_available'])
        self.assertEqual(state['face_stress_reason'], 'the camera was off for this turn')
        self.assertFalse(context['facial_stress']['available'])

    def test_face_check_is_patient_only(self):
        face = Mock()
        face.detect_face.return_value = True
        with patch.object(api, 'get_facial_analyzer', return_value=face):
            self.login(self.doctor)
            frame = {'frame': ('f.jpg', b'jpeg', 'image/jpeg')}
            self.assertEqual(self.client.post('/api/face/check', files=frame).status_code, 403)
            self.login()
            response = self.client.post('/api/face/check', files=frame)
            self.assertEqual(response.json(), {'face_detected': True})
            self.assertEqual(self.client.post('/api/face/check', files={'frame': ('a.wav', b'x', 'audio/wav')}).status_code, 400)

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
