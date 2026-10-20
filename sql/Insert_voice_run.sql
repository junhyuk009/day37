-- 실행 이력 한 건 저장 (파이썬에서 meta 딕셔너리를 그대로 전달)
-- 주의: 이 파일의 주석에는 퍼센트 기호를 쓰지 않음 (psycopg 파라미터 기호와 충돌)
INSERT INTO practice_voice_runs
    (request_id, created_at, status, stt_ms, reply_ms,
     tts_ms, total_ms, model_version, audio_sha256)
VALUES
    (%(request_id)s, %(created_at)s, %(status)s, %(stt_ms)s, %(reply_ms)s,
     %(tts_ms)s, %(total_ms)s, %(model_version)s, %(audio_sha256)s);