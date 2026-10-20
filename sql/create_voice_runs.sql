-- 음성 도우미 실행 이력 테이블 (최초 1회 실행)
CREATE TABLE IF NOT EXISTS practice_voice_runs (
    request_id    TEXT PRIMARY KEY,           -- 요청ID
    created_at    TIMESTAMPTZ NOT NULL,       -- 생성일시 (타임존)
    status        TEXT NOT NULL,              -- 상태문자열 (success / error)
    stt_ms        DOUBLE PRECISION NOT NULL,  -- STT 소요시간
    reply_ms      DOUBLE PRECISION NOT NULL,  -- 응답 소요시간
    tts_ms        DOUBLE PRECISION NOT NULL,  -- TTS 소요시간
    total_ms      DOUBLE PRECISION NOT NULL,  -- 소요시간 합계
    model_version TEXT NOT NULL,              -- 모델 버젼 문자열
    audio_sha256  TEXT NOT NULL               -- 오디오 해시값
);